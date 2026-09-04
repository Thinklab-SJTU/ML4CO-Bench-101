from ml4co_kit import TSPTask
from typing import List, Tuple
from ml4co.fast_t2t.common import MetaEnv
from ml4co.fast_t2t.common import MetaDataBatch
from ml4co.fast_t2t.tsp.tsp_dataset import TSPDataset, tsp_data_process


class TSPEnv(MetaEnv):
    def __init__(
        self,
        mode: str = None,
        train_batch_size: int = 128,
        val_batch_size: int = 128,
        train_dataset: TSPDataset = None,
        val_dataset: TSPDataset = None,
        test_dataset: TSPDataset = None,
        num_workers: int = 4,
        device: str = "cuda",
    ):
        # Super Args
        super(TSPEnv, self).__init__(
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            num_workers=num_workers,
            device=device
        )

    def process_batch_data(
        self, batch_task_data: List[TSPTask], runs_num: int = 1
    ) -> Tuple[List[TSPTask], MetaDataBatch]:
        # Process batch data
        processed_data_list = list()
        for task_data in batch_task_data:
            processed_data = tsp_data_process(task_data)
            for _ in range(runs_num):
                processed_data_list.append(processed_data)

        # Create MetaDataBatch
        meta_data_batch = MetaDataBatch()
        meta_data_batch.from_data_list(processed_data_list)
        return batch_task_data, meta_data_batch