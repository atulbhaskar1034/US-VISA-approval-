import numpy as np
# Monkey patch numpy to restore the removed float_ alias
np.float_ = np.float64 

from us_visa.pipline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()
