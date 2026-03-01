from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from . import views
urlpatterns = [
    # --- 预测/元数据  ---
    path("meta/", views.meta),
    path("forecast/", views.forecast),

    # --- 大屏接口  ---
    path("dashboard/summary/", views.dashboard_summary),
    path("dashboard/category-stats/", views.dashboard_category_stats),
    path("dashboard/avg-price-ranking/", views.dashboard_avg_price_ranking),

    # --- 农产品库 ---
    path("products/", views.products),
    path("products/timeseries/", views.product_timeseries),

    # --- 认证接口 ---
    # 使用 .as_view() 调用类视图
    path("auth/register/", views.RegisterView.as_view()),
    path("auth/login/", views.LoginView.as_view()),
    path("auth/me/", views.AuthMeView.as_view()),

    # --- 管理员接口 ---
    path("admin/users/", views.AdminUserListView.as_view()),
    path("admin/users/update/", views.AdminUserUpdateView.as_view()),
    path("admin/users/delete/", csrf_exempt(views.AdminUserDeleteView.as_view())),

    # --- 知识图谱 ---
    path("kg/graph/", views.kg_graph),

    path("forecast/history/", views.forecast_history),
]