# Deploy DentalX Camp to Vercel

`prepare_vercel_deploy.py` assembles a self-contained Vercel Services project containing the React client,
FastAPI orchestration backend, all three model APIs, and selected inference weights. It leaves the source
model folders untouched.

From the repository root:

```powershell
python prepare_vercel_deploy.py
Set-Location <printed-output-folder>
npm install --global vercel   # only if the CLI is not installed
vercel login
vercel deploy --prod
```

Enable Vercel Services for the target project. After deployment, configure these project environment
variables and redeploy:

- `APP_ENV=production`
- `MONGODB_URI` and optional `DATABASE_NAME`
- `JWT_SECRET`: random value of at least 32 characters
- `CARIES_API_URL=https://<deployment-domain>/models/caries/analyze`
- `TOOTH_API_URL=https://<deployment-domain>/models/tooth/analyze`
- `GINGIVITIS_API_URL=https://<deployment-domain>/models/gingivitis/analyze`
- Optional `MODEL_API_KEY` (model services and backend must use the same value)
- Optional SMTP settings for e-mail reports

The APIs should respond at `/api/health` and `/models/{caries,tooth,gingivitis}/health`. Check all three
model health entries before using the application. Model weights are large and inference is CPU-bound;
Vercel plan, function size, memory, duration, and Services availability can limit this deployment. The
deployment bundle can be generated and inspected locally before uploading. This app has not been deployed
from this workspace because Vercel authentication and a working MongoDB credential are not configured.

See [Vercel Services](https://vercel.com/docs/services), [Python runtime](https://vercel.com/docs/functions/runtimes/python),
and [Function limits](https://vercel.com/docs/functions/limitations).
