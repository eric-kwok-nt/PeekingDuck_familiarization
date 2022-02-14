from typing import Any, Dict
import os
from peekingduck.pipeline.nodes.node import AbstractNode
import ntpath


class Node(AbstractNode):
    """This node takes in a DataFrame and outputs a csv file. It creates parent folder if it does not exist.
    The recording is performed either with the 'write_now' flag or when 'pipeline_end' flag is True.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        node_path = os.path.join(os.getcwd(), "src/custom_nodes/configs/output.record_df_to_csv")
        super().__init__(config, node_path=node_path, **kwargs)
        # super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """
        Args:
            inputs (dict): Dictionary with keys 'df_records', 'write_now', 'pipeline_end'.

        Returns:
            outputs (dict): Empty Dictionary.
        """
        if (inputs["write_now"] or inputs["pipeline_end"]) and (inputs["df_records"] is not None):
            self.logger.info("Saving DataFrame to CSV file!")
            base, _ = ntpath.split(self.csv_path)
            # Create make new directories if the path does not exist
            if not os.path.exists(base):
                os.makedirs(base)
                self.logger.info(f"Created folder '{base}'")
            inputs["df_records"].to_csv(self.csv_path, mode=self.write_mode)
        return {}
