import pytest

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import exceptions
from local_docs_rag_agent.evals import comparison


def test_eval_matrix_rejects_unbounded_cartesian_product() -> None:
    with pytest.raises(exceptions.ConfigurationError, match="maximum"):
        comparison.run_eval_matrix(
            config=app_config.AppConfig.from_env(),
            runtimes=["basic"],
            chunk_strategies=["fixed"],
            vector_backends=["local"],
            top_ks=list(range(1, 130)),
            chunk_sizes=[800],
            chunk_overlaps=[120],
        )
