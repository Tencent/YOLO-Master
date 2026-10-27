# YOLO-Master Studio

Studio is an asynchronous task platform for model training and inference
workflows. It manages training, validation, prediction, export and diagnostic
tasks through a shared HTTP API. React, the built-in console and Gradio Jobs
are clients of this platform.

Each submitted task is a **job** that runs in the background. You can follow its
status, read logs and download its output files.

## Start from a source checkout

Use Python 3.10 or later. From the repository root, install and start Studio:

```bash
python -m pip install -e ".[studio]"
python main_engine.py
```

Keep the terminal open and visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/). This built-in console
runs directly from the checkout; it needs neither Node.js nor Gradio. The same
service provides interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The service binds to the local machine. See [security](security.md) for deployment
settings and access rules.

## Submit and inspect a job

Try an environment diagnostic first; it needs no model or dataset:

1. Check **Engine** is `http://127.0.0.1:8000` and select **Apply**.
2. Choose **Task Type → diagnose**. Leave model and data fields empty.
3. Open **Advanced options**, set **Output Directory** to `runs/diagnose`,
   and leave **Extra Params** empty.
4. Submit the job and select it in the job list. Wait for `completed`, then
   read its logs and download the JSON and text reports from the artifact list.

An **artifact** is a downloadable output file. Diagnostic reports contain local
environment information. If the job fails, read its error and logs before resubmitting.

Stop the service with Ctrl+C in its terminal. Closing the browser keeps it running.

## Next steps

- [User guide](user_guide.md): choose a client, prepare model/data paths,
  track jobs, cancel work and download results.
- [Clients and launcher](frontend.md): React development, Gradio startup and
  process ownership.
- [API](api.md): request examples, errors, log cursors, downloads and the
  [generated input schema](job_schema.json).
- [Service configuration](service.md): trusted roots, local origins and settings.
- [Architecture](architecture.md), [Core](core.md) and [Runtime](runtime.md):
  execution layers, handlers, persistence and checkpoint recovery.
- [Security](security.md): supported deployment and file-access rules.
- [Testing and reproduction](testing.md): automated checks and manual procedures.
