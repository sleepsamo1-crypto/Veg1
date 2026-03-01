# api/views.py
from __future__ import annotations

import json
import datetime
from pathlib import Path
from datetime import timedelta

import pandas as pd
import jwt

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.db.models import Avg, Min, Max
from django.utils.timezone import now
from django.contrib.auth import authenticate
from django.contrib.auth.models import User

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from api.models import VegetablePrice, PredictResult

# ---- Neo4j ----
from neo4j import GraphDatabase

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

# 你的项目根目录：D:\Veg
BASE_DIR = Path(__file__).resolve().parent.parent
BY_VEG_DIR = BASE_DIR / "saved_models" / "by_veg"

# Neo4j driver（全局复用）
_driver = GraphDatabase.driver(
    settings.NEO4J_URI,
    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
)


def _read_top3(veg: str):
    p = BY_VEG_DIR / veg / "top3.json"
    if not p.exists():
        raise FileNotFoundError(f"top3.json 不存在：{p}，请先 python -m ml.train_all 训练")
    payload = json.loads(p.read_text(encoding="utf-8"))
    top3 = payload.get("top3", [])
    models = [x["model"] for x in top3 if "model" in x]
    return payload, top3, models


def _aggregate_all_markets(veg: str) -> pd.DataFrame:
    """把一个蔬菜的 3 市场数据聚合成 per_veg 的日序列（与你训练口径一致）"""
    qs = (VegetablePrice.objects
          .filter(vegetable_name=veg)
          .values("date", "avg_price", "min_price", "max_price", "market_name")
          .order_by("date"))
    rows = list(qs)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["avg_price", "min_price", "max_price"]:
        if c in df.columns:
            df[c] = df[c].astype(float)

    out = (df.groupby("date", as_index=False)
             .agg(avg_price=("avg_price", "mean"),
                  min_price=("min_price", "min"),
                  max_price=("max_price", "max"))
             .sort_values("date")
             .reset_index(drop=True))
    out["vegetable_name"] = veg
    return out[["vegetable_name", "date", "avg_price", "min_price", "max_price"]]


def _forecast_with_top3(df_hist: pd.DataFrame, veg: str, horizon: int):
    """调用 ml/predict_entry.py 的 Top3 预测"""
    from ml.predict_entry import forecast_top3  # noqa

    hist_dates, hist_prices, future_dates, preds_by_model = forecast_top3(
        df_hist=df_hist[["vegetable_name", "date", "avg_price"]],
        veg=veg,
        horizon=horizon,
    )
    return hist_dates, hist_prices, future_dates, preds_by_model


def get_user_from_token(request):
    """兼容 HttpRequest/DRF Request：从 Authorization 解析用户"""
    auth = (request.META.get("HTTP_AUTHORIZATION") or "").strip()
    if not auth:
        auth = (getattr(request, "headers", {}).get("Authorization") or "").strip()
    if not auth:
        return None

    token = auth.split()[-1].strip().strip("'").strip('"')
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return User.objects.get(id=payload["user_id"])
    except Exception:
        return None


@require_GET
def meta(request):
    """给前端下拉框用：蔬菜列表、市场列表、veg->markets 映射、每个 veg 的可用模型"""
    vegs = (VegetablePrice.objects.values_list("vegetable_name", flat=True)
            .order_by("vegetable_name").distinct())
    markets = (VegetablePrice.objects.values_list("market_name", flat=True)
               .order_by("market_name").distinct())

    pairs = VegetablePrice.objects.values("vegetable_name", "market_name").distinct()
    veg_market_map = {}
    for p in pairs:
        veg_market_map.setdefault(p["vegetable_name"], set()).add(p["market_name"])
    veg_market_map = {k: sorted(list(v)) for k, v in veg_market_map.items()}

    veg_models = {}
    if BY_VEG_DIR.exists():
        for d in BY_VEG_DIR.iterdir():
            if d.is_dir():
                try:
                    _, _, models = _read_top3(d.name)
                    veg_models[d.name] = models
                except Exception:
                    pass

    return JsonResponse({
        "status": "success",
        "vegetables": list(vegs),
        "markets": list(markets),
        "veg_market_map": veg_market_map,
        "veg_models": veg_models,
    }, json_dumps_params={"ensure_ascii": False})


@require_GET
def forecast(request):
    """
    ✅ 预测接口：按用户每次预测入库 + 同参数命中直接复用
    缓存键：user + veg + horizon + model_version + chosen_model
    """
    veg = request.GET.get("veg", "").strip()
    market = request.GET.get("market", "ALL").strip()
    horizon = int(request.GET.get("horizon", "7"))
    decimals = int(request.GET.get("decimals", "2"))
    requested_model = (request.GET.get("model", "").strip() or None)

    if not veg:
        return JsonResponse({"status": "error", "message": "缺少 veg"}, status=400)

    # 只支持 ALL（聚合口径）
    if market.upper() != "ALL" and market != "":
        return JsonResponse({
            "status": "error",
            "message": "当前预测模型按三市场聚合训练，请选择 market=ALL(聚合)"
        }, status=400)

    # 读取 Top3 模型列表
    try:
        _, top3_metrics, top3_models = _read_top3(veg)
    except FileNotFoundError as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

    chosen = requested_model or (top3_models[0] if top3_models else None)
    if not chosen:
        return JsonResponse({"status": "error", "message": "该蔬菜没有可用模型，请先训练"}, status=500)

    # 模型版本：top3.json 修改时间
    top3_path = BY_VEG_DIR / veg / "top3.json"
    model_version = str(top3_path.stat().st_mtime_ns)

    user = get_user_from_token(request)

    # 1) 查缓存（把 chosen 也作为键，避免用户切模型时拿到旧展示结果）
    if user:
        cached = (PredictResult.objects
                  .filter(user=user,
                          vegetable_name=veg,
                          predict_days=horizon,
                          model_name=chosen,                 # ✅ 这里存“展示模型名”
                          model_version=model_version)
                  .order_by("-created_time")
                  .first())
        if cached:
            data = cached.result_json
            data["status"] = "success"
            data["source"] = "cache"
            return JsonResponse(data, json_dumps_params={"ensure_ascii": False})

    # 2) 计算预测
    df_hist = _aggregate_all_markets(veg)
    if df_hist.empty:
        return JsonResponse({"status": "error", "message": "该蔬菜无数据"}, status=404)

    hist_dates, hist_prices, future_dates, preds_by_model = _forecast_with_top3(
        df_hist=df_hist,
        veg=veg,
        horizon=horizon,
    )

    if chosen not in preds_by_model:
        return JsonResponse({
            "status": "error",
            "message": f"model={chosen} 不在该蔬菜 Top3 中，可选：{list(preds_by_model.keys())}"
        }, status=400)

    future_prices = preds_by_model.get(chosen, [])

    hist_prices = [round(float(x), decimals) for x in hist_prices]
    future_prices = [round(float(x), decimals) for x in future_prices]

    resp = {
        "veg": veg,
        "market": "ALL",
        "horizon": horizon,
        "decimals": decimals,
        "model_name": chosen,
        "model_version": model_version,
        "available_models": list(preds_by_model.keys()),
        "top3": top3_metrics,
        "dates": hist_dates,
        "real_price": hist_prices,
        "future_dates": future_dates,
        "future_prices": future_prices,
        "preds_by_model": preds_by_model,  # 仍保留，前端可画多条线
    }

    # 3) 入库（按用户每次预测存一条）
    if user:
        try:
            predict_start = (pd.to_datetime(hist_dates[-1]) + timedelta(days=1)).date() if hist_dates else None
        except Exception:
            predict_start = None

        PredictResult.objects.create(
            user=user,
            vegetable_name=veg,
            predict_days=horizon,
            model_name=chosen,               # ✅ 存展示模型名（与缓存键一致）
            model_version=model_version,
            predict_start_date=predict_start,
            result_json=resp,
        )

    resp["status"] = "success"
    resp["source"] = "compute"
    return JsonResponse(resp, json_dumps_params={"ensure_ascii": False})


# ------------------------- 下面保持你原逻辑不变 -------------------------

VEG_CATEGORY_MAP = {
    "白菜": "叶菜类", "小白菜": "叶菜类", "菠菜": "叶菜类", "生菜": "叶菜类", "油麦菜": "叶菜类",
    "土豆": "根茎类", "萝卜": "根茎类", "胡萝卜": "根茎类",
    "番茄": "茄果类", "西红柿": "茄果类", "茄子": "茄果类", "辣椒": "茄果类",
    "黄瓜": "瓜果类", "冬瓜": "瓜果类", "南瓜": "瓜果类",
    "大葱": "葱蒜类", "生姜": "葱蒜类", "大蒜": "葱蒜类",
    "香菇": "菌菇类", "金针菇": "菌菇类",
    "豆角": "豆类", "四季豆": "豆类",
}

def veg_to_category(veg: str) -> str:
    return VEG_CATEGORY_MAP.get(veg, "其他")


@require_GET
def dashboard_summary(request):
    total_records = VegetablePrice.objects.count()
    veg_cnt = VegetablePrice.objects.values("vegetable_name").distinct().count()
    market_cnt = VegetablePrice.objects.values("market_name").distinct().count()

    agg = VegetablePrice.objects.aggregate(
        min_date=Min("date"),
        max_date=Max("date"),
        max_crawl=Max("crawl_time"),
    )

    return JsonResponse({
        "status": "success",
        "total_records": total_records,
        "vegetable_count": veg_cnt,
        "market_count": market_cnt,
        "min_date": agg["min_date"].isoformat() if agg["min_date"] else None,
        "max_date": agg["max_date"].isoformat() if agg["max_date"] else None,
        "last_crawl_time": agg["max_crawl"].isoformat() if agg["max_crawl"] else None,
    }, json_dumps_params={"ensure_ascii": False})


@require_GET
def dashboard_category_stats(request):
    window = int(request.GET.get("window", "30"))

    agg = VegetablePrice.objects.aggregate(max_date=Max("date"))
    end_date = agg["max_date"] or now().date()
    start_date = end_date - timedelta(days=window - 1)

    qs = (VegetablePrice.objects
          .filter(date__gte=start_date, date__lte=end_date)
          .values("vegetable_name", "avg_price", "date"))

    rows = list(qs)
    if not rows:
        return JsonResponse({"status": "success", "window": window, "categories": []},
                            json_dumps_params={"ensure_ascii": False})

    df = pd.DataFrame(rows)
    df["avg_price"] = df["avg_price"].astype(float)
    df["category"] = df["vegetable_name"].apply(veg_to_category)

    veg_avg = (df.groupby(["category", "vegetable_name"], as_index=False)
               .agg(veg_avg_price=("avg_price", "mean")))

    out = (veg_avg.groupby("category", as_index=False)
           .agg(veg_count=("vegetable_name", "nunique"),
                avg_price=("veg_avg_price", "mean"))
           .sort_values("veg_count", ascending=False))

    return JsonResponse({
        "status": "success",
        "window": window,
        "categories": out.to_dict(orient="records"),
    }, json_dumps_params={"ensure_ascii": False})


@require_GET
def dashboard_avg_price_ranking(request):
    window = int(request.GET.get("window", "30"))
    topn = int(request.GET.get("top", "10"))

    agg = VegetablePrice.objects.aggregate(max_date=Max("date"))
    end_date = agg["max_date"] or now().date()
    start_date = end_date - timedelta(days=window - 1)

    qs = (VegetablePrice.objects
          .filter(date__gte=start_date, date__lte=end_date)
          .values("vegetable_name")
          .annotate(avg_price=Avg("avg_price"))
          .order_by("-avg_price")[:topn])

    data = []
    for r in qs:
        veg = r["vegetable_name"]
        data.append({
            "vegetable_name": veg,
            "category": veg_to_category(veg),
            "avg_price": round(float(r["avg_price"]), 4) if r["avg_price"] is not None else None,
        })

    return JsonResponse({
        "status": "success",
        "window": window,
        "top": topn,
        "ranking": data,
    }, json_dumps_params={"ensure_ascii": False})


@require_GET
def products(request):
    q = (request.GET.get("q", "") or "").strip()
    category = (request.GET.get("category", "") or "").strip()
    sort = (request.GET.get("sort", "price_desc") or "").strip()
    page = int(request.GET.get("page", "1"))
    page_size = int(request.GET.get("page_size", "12"))

    qs = (VegetablePrice.objects
          .values("vegetable_name", "market_name", "date", "avg_price", "min_price", "max_price"))

    rows = list(qs)
    if not rows:
        return JsonResponse({"status": "success", "total": 0, "items": []},
                            json_dumps_params={"ensure_ascii": False})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["avg_price", "min_price", "max_price"]:
        df[c] = df[c].astype(float)

    latest_date = df.groupby("vegetable_name")["date"].max().reset_index().rename(columns={"date": "latest_date"})
    df = df.merge(latest_date, on="vegetable_name", how="left")
    df_latest = df[df["date"] == df["latest_date"]].copy()

    agg = (df_latest.groupby(["vegetable_name", "latest_date"], as_index=False)
           .agg(avg_price=("avg_price", "mean"),
                min_price=("min_price", "min"),
                max_price=("max_price", "max")))
    agg["category"] = agg["vegetable_name"].apply(veg_to_category)
    agg["date"] = agg["latest_date"].dt.date.astype(str)

    if q:
        agg = agg[agg["vegetable_name"].str.contains(q, na=False)]
    if category:
        agg = agg[agg["category"] == category]

    if sort == "price_asc":
        agg = agg.sort_values("avg_price", ascending=True)
    elif sort == "name_asc":
        agg = agg.sort_values("vegetable_name", ascending=True)
    else:
        agg = agg.sort_values("avg_price", ascending=False)

    total = len(agg)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = agg.iloc[start:end].copy()

    items = []
    for _, r in page_df.iterrows():
        items.append({
            "vegetable_name": r["vegetable_name"],
            "category": r["category"],
            "date": r["date"],
            "avg_price": round(float(r["avg_price"]), 4),
            "min_price": round(float(r["min_price"]), 4),
            "max_price": round(float(r["max_price"]), 4),
        })

    return JsonResponse({
        "status": "success",
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": items,
    }, json_dumps_params={"ensure_ascii": False})


@require_GET
def product_timeseries(request):
    veg = (request.GET.get("veg", "") or "").strip()
    window = int(request.GET.get("window", "30"))
    if not veg:
        return JsonResponse({"status": "error", "message": "缺少 veg"}, status=400)

    agg = VegetablePrice.objects.filter(vegetable_name=veg).aggregate(max_date=Max("date"))
    end_date = agg["max_date"] or now().date()
    start_date = end_date - timedelta(days=window - 1)

    qs = (VegetablePrice.objects
          .filter(vegetable_name=veg, date__gte=start_date, date__lte=end_date)
          .values("date", "avg_price", "min_price", "max_price")
          .order_by("date"))
    rows = list(qs)
    if not rows:
        return JsonResponse(
            {"status": "success", "veg": veg, "window": window, "dates": [], "avg": [], "min": [], "max": []},
            json_dumps_params={"ensure_ascii": False}
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["avg_price", "min_price", "max_price"]:
        df[c] = df[c].astype(float)

    out = (df.groupby("date", as_index=False)
           .agg(avg=("avg_price", "mean"), min=("min_price", "min"), max=("max_price", "max"))
           .sort_values("date"))

    return JsonResponse({
        "status": "success",
        "veg": veg,
        "window": window,
        "end_date": str(end_date),
        "dates": out["date"].dt.date.astype(str).tolist(),
        "avg": [round(float(x), 4) for x in out["avg"].tolist()],
        "min": [round(float(x), 4) for x in out["min"].tolist()],
        "max": [round(float(x), 4) for x in out["max"].tolist()],
    }, json_dumps_params={"ensure_ascii": False})


# 注册
class RegisterView(APIView):
    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")
        role = request.data.get("role", "user")

        if not username or not password:
            return Response({"message": "用户名和密码不能为空"}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(username=username).exists():
            return Response({"message": "用户名已存在"}, status=status.HTTP_400_BAD_REQUEST)

        user = User.objects.create_user(username=username, password=password)
        user.is_staff = (role == "admin")
        user.save()

        return Response({"status": "success", "message": "注册成功"}, status=status.HTTP_201_CREATED)


# 登录
class LoginView(APIView):
    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")

        user = authenticate(username=username, password=password)
        if user is not None:
            token = jwt.encode({
                "user_id": user.id,
                "username": user.username,
                "is_staff": user.is_staff,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
            }, settings.SECRET_KEY, algorithm="HS256")

            if isinstance(token, bytes):
                token = token.decode("utf-8")

            return Response({
                "status": "success",
                "message": "登录成功",
                "token": token,
                "username": user.username,
                "role": "admin" if user.is_staff else "user"
            }, status=status.HTTP_200_OK)

        return Response({"status": "error", "message": "用户名或密码错误"}, status=status.HTTP_401_UNAUTHORIZED)


# 获取个人信息
class AuthMeView(APIView):
    def get(self, request):
        user = get_user_from_token(request)
        if not user:
            return Response({"message": "无效的token"}, status=status.HTTP_401_UNAUTHORIZED)
        return Response({
            "username": user.username,
            "role": "admin" if user.is_staff else "user"
        })

from rest_framework.exceptions import PermissionDenied

class AdminBaseView(APIView):
    def initial(self, request, *args, **kwargs):
        super().initial(request, *args, **kwargs)
        user = get_user_from_token(request)
        if not user or not user.is_staff:
            raise PermissionDenied("管理员权限不足")
        request.user = user
class AdminUserListView(AdminBaseView):
    def get(self, request):
        users = User.objects.all().values("id", "username", "is_staff", "is_active")
        return Response({"items": list(users)})


class AdminUserUpdateView(AdminBaseView):
    def post(self, request):
        user_id = request.data.get("id")
        role = request.data.get("role")
        is_active = request.data.get("is_active")

        try:
            user = User.objects.get(id=user_id)
            if role:
                user.is_staff = (role == "admin")
            if is_active is not None:
                user.is_active = is_active
            user.save()
            return Response({"message": "用户信息更新成功"})
        except User.DoesNotExist:
            return Response({"message": "用户不存在"}, status=status.HTTP_404_NOT_FOUND)


class AdminUserDeleteView(AdminBaseView):
    def post(self, request):
        user_id = request.data.get("id")
        try:
            user = User.objects.get(id=user_id)
            user.delete()
            return Response({"message": "用户删除成功"})
        except User.DoesNotExist:
            return Response({"message": "用户不存在"}, status=status.HTTP_404_NOT_FOUND)


# ---- 知识图谱 ----
def kg_graph(request):
    limit = int(request.GET.get("limit", 500))
    cypher = """
    MATCH (a:Entity)-[r:REL]->(b:Entity)
    RETURN a.name AS source, r.predicate AS predicate, b.name AS target
    LIMIT $limit
    """
    try:
        with _driver.session() as session:
            rows = session.run(cypher, limit=limit).data()
    except Exception as e:
        return JsonResponse({"status": "error", "message": f"Neo4j 查询失败：{e}"}, status=500)

    nodes_set = set()
    links = []
    for row in rows:
        s = row["source"]
        t = row["target"]
        p = row.get("predicate") or ""
        nodes_set.add(s)
        nodes_set.add(t)
        links.append({"source": s, "target": t, "name": p})

    nodes = [{"id": n, "name": n} for n in nodes_set]
    return JsonResponse({"nodes": nodes, "links": links}, json_dumps_params={"ensure_ascii": False})


# ---- 预测历史 ----
@require_GET
def forecast_history(request):
    user = get_user_from_token(request)
    if not user:
        return JsonResponse({"status": "error", "message": "未登录"}, status=401)

    limit = int(request.GET.get("limit", "20"))
    veg = (request.GET.get("veg", "") or "").strip()

    qs = PredictResult.objects.filter(user=user)
    if veg:
        qs = qs.filter(vegetable_name=veg)

    qs = qs.order_by("-created_time")[:limit]

    items = [{
        "id": r.id,
        "vegetable_name": r.vegetable_name,
        "predict_days": r.predict_days,
        "model_name": r.model_name,
        "model_version": r.model_version,
        "created_time": r.created_time.isoformat(),
    } for r in qs]

    return JsonResponse({"status": "success", "items": items}, json_dumps_params={"ensure_ascii": False})