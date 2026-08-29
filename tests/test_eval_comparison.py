import pytest

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.evals.comparison import run_eval_matrix
from local_docs_rag_agent.exceptions import ConfigurationError


def test_eval_matrix_rejects_unbounded_cartesian_product() -> None:
    with pytest.raises(ConfigurationError, match="maximum"):
        run_eval_matrix(
            config=AppConfig.from_env(),
            runtimes=["basic"],
            chunk_strategies=["fixed"],
            vector_backends=["local"],
            top_ks=list(range(1, 130)),
            chunk_sizes=[800],
            chunk_overlaps=[120],
        )
