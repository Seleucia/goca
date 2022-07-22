import enum

class ChannelModal(enum.Enum):
    rgb = 'rgb'
    flow = 'flow'
    rgb_flow = 'rgb_flow'

class EvalType(enum.Enum):
    PreTrain = 0
    LinCls = 1
    FT = 2
    KNN_emb_prot = 3
    KNN_emb = 4
    KNN_all = 5
    KNN_FullOpt=[3,4,5]

class ModalMergeType(enum.Enum):
    Avg = 0
    Concat = 1
    ProcSeperate = 2


class ModalMergeAlgo(enum.Enum):
    AvgRgbFlow = 0
    MixRgbFlow = 1
    PriorRgbFlow = 2


