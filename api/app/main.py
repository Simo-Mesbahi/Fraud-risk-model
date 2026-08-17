from __future__ import annotations

import logging
import os
import time
import uuid

from contextlib import asynccontextmanager
from typing import (
    AsyncIterator,
    Final,
)

from fastapi import (
    FastAPI,
    Request,
    status,
)

from fastapi.exceptions import (
    RequestValidationError,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.responses import (
    JSONResponse,
)

from api.app.dependencies import (
    get_fraud_scorer,
    get_prediction_service,
    validate_model_artifacts,
)

from api.app.routers.healthcheck import (
    router as healthcheck_router,
)

from api.app.routers.predictions import (
    router as predictions_router,
)


# =============================================================================
# Application metadata
# =============================================================================


APP_NAME: Final[str] = (
    "Health Insurance Fraud Risk API"
)

APP_VERSION: Final[str] = (
    "1.1.0"
)

APP_DESCRIPTION: Final[str] = """
Production-style inference API for AI-assisted health-insurance fraud
investigation prioritization.

### Capabilities

The service exposes:

- single-claim fraud-risk scoring;
- portfolio batch scoring;
- investigation-capacity prioritization;
- local TreeSHAP explanations;
- deployed-model metadata;
- runtime health information.

### Decision-support policy

The predicted fraud-risk score supports human investigation prioritization.

It does **not** establish that fraud occurred and must not be used as an
automatic claim-rejection or fraud-adjudication mechanism.

### Explainability

Local explanations are generated from the same frozen preprocessing pipeline
and XGBoost model used for inference.

TreeSHAP contributions are expressed in raw model-margin/log-odds space and
are numerically checked against the deployed model output.
"""


# =============================================================================
# Runtime configuration
# =============================================================================


DEFAULT_LOG_LEVEL: Final[str] = (
    "INFO"
)

DEFAULT_ENVIRONMENT: Final[str] = (
    "development"
)


# =============================================================================
# Logging
# =============================================================================


def _resolve_log_level() -> int:
    """
    Resolve the configured logging level safely.

    Invalid LOG_LEVEL values fall back to INFO instead of preventing
    application startup.
    """

    configured = (
        os.getenv(
            "LOG_LEVEL",
            DEFAULT_LOG_LEVEL,
        )
        .strip()
        .upper()
    )

    resolved = getattr(
        logging,
        configured,
        None,
    )

    if not isinstance(
        resolved,
        int,
    ):
        return logging.INFO

    return resolved


logging.basicConfig(
    level=_resolve_log_level(),
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)


logger = logging.getLogger(
    "health_fraud.api"
)


# =============================================================================
# Environment helpers
# =============================================================================


def _environment_name() -> str:
    """
    Return the normalized runtime environment name.
    """

    value = (
        os.getenv(
            "APP_ENV",
            DEFAULT_ENVIRONMENT,
        )
        .strip()
        .lower()
    )

    return (
        value
        or DEFAULT_ENVIRONMENT
    )


def _cors_origins() -> list[str]:
    """
    Resolve explicitly allowed browser origins.

    Example
    -------
    CORS_ORIGINS=https://app.example.com,https://admin.example.com

    In development, common local Streamlit origins are enabled when
    CORS_ORIGINS is not explicitly configured.
    """

    configured = (
        os.getenv(
            "CORS_ORIGINS"
        )
    )

    if configured:

        origins = [
            origin.strip()
            for origin
            in configured.split(",")
            if origin.strip()
        ]

        # Preserve order while removing duplicates.
        return list(
            dict.fromkeys(
                origins
            )
        )

    if (
        _environment_name()
        == "development"
    ):

        return [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ]

    return []


# =============================================================================
# Application lifecycle
# =============================================================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    Manage inference-service startup and shutdown.

    Startup is deliberately fail-fast:

    1. validate persisted model artifacts;
    2. deserialize the frozen inference stack;
    3. initialize the prediction service;
    4. validate the deployed model contract;
    5. expose the API only after successful initialization.

    A broken or incomplete model deployment therefore cannot silently
    present itself as a healthy inference service.
    """

    startup_started = (
        time.perf_counter()
    )

    environment = (
        _environment_name()
    )

    logger.info(
        "Starting %s v%s",
        APP_NAME,
        APP_VERSION,
    )

    logger.info(
        "Runtime environment: %s",
        environment,
    )

    try:

        # ---------------------------------------------------------------------
        # 1. Validate persisted artifacts
        # ---------------------------------------------------------------------

        validate_model_artifacts()

        logger.info(
            "Model artifacts validated."
        )

        # ---------------------------------------------------------------------
        # 2. Warm the frozen ML inference stack
        # ---------------------------------------------------------------------

        scorer = (
            get_fraud_scorer()
        )

        # ---------------------------------------------------------------------
        # 3. Warm the application service
        # ---------------------------------------------------------------------

        service = (
            get_prediction_service()
        )

        # ---------------------------------------------------------------------
        # 4. Validate deployed model information
        # ---------------------------------------------------------------------

        info = (
            service.model_info()
        )

        model_name = str(
            info.get(
                "model_name",
                "unknown",
            )
        )

        model_version = str(
            info.get(
                "model_version",
                "unknown",
            )
        )

        source_feature_count = (
            info.get(
                "feature_count",
                "unknown",
            )
        )

        transformed_feature_count = (
            info.get(
                "transformed_feature_count"
            )
        )

        if (
            transformed_feature_count
            is None
        ):

            transformed_feature_count = len(
                getattr(
                    scorer,
                    "transformed_feature_names",
                    [],
                )
            )

        startup_ms = (
            (
                time.perf_counter()
                - startup_started
            )
            * 1000.0
        )

        logger.info(
            (
                "Inference stack ready | "
                "model=%s | "
                "version=%s | "
                "source_features=%s | "
                "transformed_features=%s | "
                "startup_ms=%.1f"
            ),
            model_name,
            model_version,
            source_feature_count,
            transformed_feature_count,
            startup_ms,
        )

        # ---------------------------------------------------------------------
        # 5. Explainability capability
        # ---------------------------------------------------------------------

        explainability = (
            info.get(
                "explainability"
            )
        )

        if isinstance(
            explainability,
            dict,
        ):

            logger.info(
                (
                    "Explainability ready | "
                    "available=%s | "
                    "method=%s | "
                    "space=%s | "
                    "transformed_features=%s"
                ),
                explainability.get(
                    "available"
                ),
                explainability.get(
                    "method"
                ),
                explainability.get(
                    "output_space"
                ),
                explainability.get(
                    "transformed_feature_count",
                    transformed_feature_count,
                ),
            )

        else:

            logger.warning(
                (
                    "No explainability metadata "
                    "reported by deployed model."
                )
            )

        # ---------------------------------------------------------------------
        # Expose useful runtime metadata internally
        # ---------------------------------------------------------------------

        app.state.started_at_monotonic = (
            time.monotonic()
        )

        app.state.environment = (
            environment
        )

        app.state.model_name = (
            model_name
        )

        app.state.model_version = (
            model_version
        )

        app.state.source_feature_count = (
            source_feature_count
        )

        app.state.transformed_feature_count = (
            transformed_feature_count
        )

        app.state.inference_ready = (
            True
        )

    except Exception:

        app.state.inference_ready = (
            False
        )

        logger.exception(
            (
                "Inference-service startup failed. "
                "The API will not start."
            )
        )

        raise

    try:

        yield

    finally:

        app.state.inference_ready = (
            False
        )

        logger.info(
            "Shutting down %s.",
            APP_NAME,
        )


# =============================================================================
# FastAPI application
# =============================================================================


app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,

    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",

    contact={
        "name":
            "Health Fraud Intelligence",
    },

    license_info={
        "name":
            "Internal decision-support prototype",
    },

    openapi_tags=[
        {
            "name":
                "system",

            "description":
                (
                    "Runtime health and deployed-model "
                    "contract information."
                ),
        },
        {
            "name":
                "scoring",

            "description":
                (
                    "Fraud-risk scoring, portfolio "
                    "prioritization and local "
                    "model explainability."
                ),
        },
    ],
)


# =============================================================================
# CORS
# =============================================================================


allowed_origins = (
    _cors_origins()
)


if allowed_origins:

    app.add_middleware(
        CORSMiddleware,

        allow_origins=(
            allowed_origins
        ),

        allow_credentials=False,

        allow_methods=[
            "GET",
            "POST",
        ],

        allow_headers=[
            "Accept",
            "Content-Type",
            "X-Request-ID",
        ],

        expose_headers=[
            "X-Request-ID",
            "X-Process-Time-Ms",
        ],
    )


# =============================================================================
# Request context middleware
# =============================================================================


@app.middleware(
    "http"
)
async def request_context_middleware(
    request: Request,
    call_next,
):
    """
    Attach operational metadata to every HTTP response.

    Adds:
    - a correlation/request identifier;
    - request processing time;
    - conservative browser-security headers.

    The request identifier can be supplied by an upstream caller through
    X-Request-ID or generated by this API.
    """

    request_id = (
        request.headers.get(
            "X-Request-ID"
        )
        or uuid.uuid4().hex
    )

    request.state.request_id = (
        request_id
    )

    started = (
        time.perf_counter()
    )

    try:

        response = await call_next(
            request
        )

    except Exception:

        elapsed_ms = (
            (
                time.perf_counter()
                - started
            )
            * 1000.0
        )

        logger.exception(
            (
                "Unhandled request failure | "
                "request_id=%s | "
                "method=%s | "
                "path=%s | "
                "elapsed_ms=%.1f"
            ),
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )

        raise

    elapsed_ms = (
        (
            time.perf_counter()
            - started
        )
        * 1000.0
    )

    response.headers[
        "X-Request-ID"
    ] = request_id

    response.headers[
        "X-Process-Time-Ms"
    ] = f"{elapsed_ms:.2f}"

    # -------------------------------------------------------------------------
    # Browser / transport hardening
    # -------------------------------------------------------------------------

    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    response.headers[
        "X-Frame-Options"
    ] = "DENY"

    response.headers[
        "Referrer-Policy"
    ] = "no-referrer"

    response.headers[
        "Cache-Control"
    ] = "no-store"

    response.headers[
        "Permissions-Policy"
    ] = (
        "camera=(), microphone=(), geolocation=()"
    )

    logger.info(
        (
            "HTTP request | "
            "request_id=%s | "
            "method=%s | "
            "path=%s | "
            "status=%s | "
            "elapsed_ms=%.1f"
        ),
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )

    return response


# =============================================================================
# Request-validation errors
# =============================================================================


@app.exception_handler(
    RequestValidationError
)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Return a stable validation-error contract.

    Pydantic validation details are preserved so API consumers can
    identify the exact invalid field while retaining the request ID
    required for operational tracing.
    """

    request_id = (
        getattr(
            request.state,
            "request_id",
            None,
        )
    )

    logger.warning(
        (
            "Request validation failed | "
            "request_id=%s | "
            "method=%s | "
            "path=%s"
        ),
        request_id,
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=(
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ),
        content={
            "detail":
                exc.errors(),

            "request_id":
                request_id,
        },
    )


# =============================================================================
# Router registration
# =============================================================================


# Keep FastAPI's official router-registration mechanism.
#
# Do not manually append healthcheck_router.routes or
# predictions_router.routes to app.router.routes. include_router()
# preserves dependencies, metadata, response models, OpenAPI information,
# callbacks and future FastAPI compatibility.

app.include_router(
    healthcheck_router
)

app.include_router(
    predictions_router
)


# =============================================================================
# Root endpoint
# =============================================================================


@app.get(
    "/",
    tags=[
        "system",
    ],
    status_code=(
        status.HTTP_200_OK
    ),
    summary=(
        "Get API service information"
    ),
    description=(
        "Return service identity and links "
        "to the principal runtime, scoring "
        "and documentation endpoints."
    ),
    operation_id="api_root",
)
def root() -> dict[
    str,
    object,
]:
    """
    Return lightweight API discovery information.
    """

    return {
        "service":
            APP_NAME,

        "version":
            APP_VERSION,

        "status":
            "running",

        "environment":
            _environment_name(),

        "purpose":
            (
                "AI-assisted health-insurance "
                "fraud investigation prioritization"
            ),

        "decision_policy":
            (
                "Human decision support only; "
                "not automatic fraud adjudication."
            ),

        "capabilities": {
            "single_claim_scoring":
                True,

            "batch_scoring":
                True,

            "investigation_prioritization":
                True,

            "local_explainability":
                True,

            "explanation_method":
                "TreeSHAP",
        },

        "endpoints": {
            "health":
                "/health",

            "model_info":
                "/model-info",

            "score":
                "/score",

            "score_batch":
                "/score-batch",

            "top_review":
                "/top-review",

            "explain":
                "/explain",
        },

        "documentation": {
            "swagger":
                "/docs",

            "redoc":
                "/redoc",

            "openapi":
                "/openapi.json",
        },
    }