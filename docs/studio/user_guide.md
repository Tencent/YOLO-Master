# Studio user guide

Start with the [quick start](README.md) to install Studio and run a
diagnostic. This guide covers daily use after the API service is running.

## Choose a client

All three clients share the same API and job history:

- **Built-in console:** visit <http://127.0.0.1:8000/> after starting
  `python main_engine.py`. No frontend build is needed.
- **React:** with Node.js 24, run `npm ci` and `npm run dev` from `web/` in a
  second terminal. Open the local address printed by Vite, normally
  <http://127.0.0.1:5173/>. Set **Engine** to `http://127.0.0.1:8000` and **Apply**.
- **Gradio:** run `python -m pip install gradio pandas` in the active environment.
  Stop the standalone API with Ctrl+C if you want the launcher to own both services,
  then run `python start_studio.py`. Open its printed URL and select **Jobs**.
  The separate **Inference Studio** tab runs synchronous inference.

For separate Gradio/API processes, custom API addresses or a production React
bundle, see [clients and launcher](frontend.md).

## Prepare a task

Choose a task type according to the work you want to run:

- `diagnose`: collect environment reports; no model or dataset needed.
- `predict`: provide a model and an image, video, media directory or input list.
- `train`: provide a model and a dataset YAML file, then set training options.
- `val`: provide a model and a dataset YAML file to measure validation metrics.
- `export`: provide a model and select an export format.

Paths refer to files on the API machine. Use existing model weights inside the
server's model roots, input data inside its data roots, and an output directory
inside its output root (`runs` by default). These **roots** are directories the
operator permits Studio to access. Model/data roots default to the repository
root when started there. Ask the operator to configure them when
needed; client path fields cannot change those permissions. See
[service settings](service.md#trusted-configuration-and-existing-ui-entry-points).

For a first model task, choose `cpu` if CUDA is unavailable. For a diagnostic,
leave model/data fields and extra parameters empty and use `runs/diagnose` as
the output directory. Fields and defaults vary by client; the
[API guide](api.md) shows the corresponding request format.

## Follow progress and cancel

After submission, select the job in the list and follow its status and logs:

- `pending`: waiting for an execution slot.
- `running`: executing, or finishing cleanup after a stop request.
- `completed`: execution and cleanup finished successfully.
- `failed`: execution or service shutdown ended the job; inspect its error code.
- `cancelled`: cancellation finished and the worker was cleaned up.

**Cancel** requests a stop. Keep checking until the job reaches a final status.
A missing-job message means the selected API cannot find that ID; check the
Engine address and job ID.

## Download results

Select a file in the job's artifact list to preview or download it. Nested names
such as `weights/shutdown.pt` identify files within that job. In Gradio,
**Open Downloads** opens the client's downloaded cache, which may differ from
the API's output directory. Files not listed as artifacts cannot be downloaded
through the API.

## Stop and recover

Keep the API terminal open while jobs run. Closing a client does not stop its
jobs; Ctrl+C requests service shutdown. The combined launcher stops services it
started and leaves an externally started API under its owner's control.

After an API restart, interrupted jobs become `failed / SERVICE_RESTARTED`.
Training does not resume automatically. A training job stopped by service
shutdown may expose `weights/shutdown.pt` while remaining
`failed / SERVICE_SHUTDOWN`. Use the [checkpoint recovery procedure](runtime.md#training-shutdown-and-recovery)
to load a trusted checkpoint and explicitly resume from the next epoch.
