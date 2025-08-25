from hrm_system.algorithms.torch_base_algorithm import TorchBaseAlgorithm


class HRMAlgorithm(TorchBaseAlgorithm):
    """
    The HRM algorithm implementation. It inherits the common training logic
    from TorchBaseAlgorithm.
    """
    
    def _init_train_state(self, train_metadata, world_size: int, rank: int):
        # Call the parent implementation
        super()._init_train_state(train_metadata, world_size, rank)
        
        # Ensure memory is disabled for HRM
        if hasattr(self.train_state.model, 'model') and hasattr(self.train_state.model.model, 'use_memory'):
            self.train_state.model.model.use_memory = False
