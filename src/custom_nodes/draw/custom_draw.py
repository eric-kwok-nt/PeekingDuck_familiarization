"""
Node template for creating custom nodes.
"""

from typing import Any, Dict
import os
from peekingduck.pipeline.nodes.node import AbstractNode
import pdb


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        node_path = os.path.join(os.getcwd(), "src/custom_nodes/configs/draw.custom_draw")
        super().__init__(config, node_path=node_path, **kwargs)
        # super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        draw_pipeline = inputs["draw_pipeline"]
        for tup in draw_pipeline:
            tup[0](**tup[1])
        inputs["draw_pipeline"] = []
        return {}
        
