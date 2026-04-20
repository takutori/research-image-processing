from __future__ import annotations

from label_studio_ml.api import init_app
from ml_backend import SAM2Backend

app = init_app(model_class=SAM2Backend)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8003, debug=False)
