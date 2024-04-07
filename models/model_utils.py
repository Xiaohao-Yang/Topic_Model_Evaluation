from models.LDA import utils as lda_utils
from models.NVDM import utils as nvdm_utils
from models.PLDA import utils as plda_utils
from models.ETM import utils as etm_utils
from models.NSTM import utils as nstm_utils


class model_utils:
    def __init__(self, model):
        if model == 'LDA':
            self.funcs = lda_utils
        elif model == 'NVDM':
            self.funcs = nvdm_utils
        elif model == 'PLDA':
            self.funcs = plda_utils
        elif model == 'ETM':
            self.funcs = etm_utils
        elif model == 'NSTM':
            self.funcs = nstm_utils


