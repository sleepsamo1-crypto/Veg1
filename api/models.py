# api/models.py
from django.conf import settings
from django.db import models
class VegetablePrice(models.Model):
    vegetable_name = models.CharField(max_length=50)
    market_name = models.CharField(max_length=100)
    min_price = models.DecimalField(max_digits=6, decimal_places=2)
    max_price = models.DecimalField(max_digits=6, decimal_places=2)
    avg_price = models.DecimalField(max_digits=6, decimal_places=2)
    date = models.DateField()
    crawl_time = models.DateTimeField()

    province_code = models.CharField(max_length=20)
    province_name = models.CharField(max_length=20)
    category = models.CharField(max_length=20)

    class Meta:
        db_table = "vegetable_price"
        managed = False

    def __str__(self):
        return f"{self.vegetable_name} - {self.market_name} - {self.date}"

class PredictResult(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="predict_results",
        null=True,
        blank=True,
    )

    vegetable_name = models.CharField(max_length=50, db_index=True)
    predict_days = models.PositiveIntegerField(default=7)

    model_name = models.CharField(max_length=50, default="top3", db_index=True)
    model_version = models.CharField(max_length=50, default="v1", db_index=True)

    predict_start_date = models.DateField(null=True, blank=True)

    result_json = models.JSONField()

    created_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["vegetable_name", "predict_days", "model_name", "model_version"]),
            models.Index(fields=["created_time"]),
        ]
        ordering = ["-created_time"]

    def __str__(self):
        return f"{self.vegetable_name}-{self.predict_days}-{self.model_name}"