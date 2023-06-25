import triton_python_backend_utils as pb_utils
from pathlib import Path
import numpy as np
import shutil
import tarfile

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    @staticmethod
    def auto_complete_config(auto_complete_model_config):

        print(auto_complete_model_config)
        return auto_complete_model_config

    def initialize(self, args):
          
        #conda env setup
        self.conda_pack_path = Path(args['model_repository']) / "sd_env.tar.gz"
        self.conda_target_path = Path("/tmp/conda")
        
        self.conda_env_path = self.conda_target_path / "sd_env.tar.gz"
             
        if not self.conda_env_path.exists():
            self.conda_env_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.conda_pack_path, self.conda_env_path)
        
        #base diffusion model setup
        self.base_model_path = Path(args['model_repository']) / "stable_diff.tar.gz"
        
        try:
            with tarfile.open(self.base_model_path) as tar:
                tar.extractall('/tmp')
                
            self.response_message = "Model env setup successful."
        
        except Exception as e:
            # print the exception message
            print(f"Caught an exception: {e}")
            self.response_message = f"Caught an exception: {e}"
        
        
    def execute(self, requests):
        
        return [pb_utils.InferenceResponse([pb_utils.Tensor("output_message", np.array(self.response_message).astype(object))])]
    

    def finalize(self):

        print('Cleaning up...')