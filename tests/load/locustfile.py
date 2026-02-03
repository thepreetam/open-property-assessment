"""
Locust load test: POST /api/v1/jobs (multipart) then poll GET /api/v1/jobs/{job_id} until completed or timeout.
Target: 12–15 jobs/hour sustained (e.g. 1 user spawning every 4–5 min) to validate ~750 assessments/week.
Run: locust -f tests/load/locustfile.py --host=http://localhost:8000
      Optional: --users 2 --spawn-rate 0.5 --run-time 1h
"""
import base64
import time
from locust import HttpUser, task, between

# Minimal 1x1 PNG (valid image for multipart upload)
TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
TINY_PNG_BYTES = base64.b64decode(TINY_PNG_B64)


class RepairOptimizerUser(HttpUser):
    """Simulates a user creating a job and polling until completion (or timeout)."""

    wait_time = between(240, 300)  # 4–5 min between jobs → ~12–15/hour per user

    @task
    def create_job_and_poll(self):
        # POST /api/v1/jobs with multipart: files + home_value + zip_code
        with self.client.post(
            "/api/v1/jobs",
            data={
                "home_value": 450000,
                "zip_code": "85001",
            },
            files=[("files", ("photo.png", TINY_PNG_BYTES, "image/png"))],
            name="/api/v1/jobs [POST]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"create job: {resp.status_code}")
                return
            try:
                data = resp.json()
                job_id = data.get("job_id")
            except Exception as e:
                resp.failure(str(e))
                return

        # Poll GET /api/v1/jobs/{job_id} until completed, failed, or timeout (e.g. 5 min)
        timeout = 300
        start = time.time()
        while time.time() - start < timeout:
            r = self.client.get(f"/api/v1/jobs/{job_id}", name="/api/v1/jobs/{id} [GET]")
            if r.status_code != 200:
                break
            try:
                st = r.json().get("status")
            except Exception:
                break
            if st in ("completed", "failed"):
                break
            time.sleep(2)
