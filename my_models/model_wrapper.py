from my_models.model_aug import load_model
import my_models.LinearEvalModels as LEM

class mwrapper(object):
    def __init__(self,args
    ):
        self.args=args
        self.video_model = load_model(args,
                                is_eval_mode_on=True,
                                )

        if self.is_eval_mode_on == True:
            self.linear_classifier = LEM.RegLog(args.nclass, args.vid_base_arch,
                                            use_lincls_use_bn=args.use_lincls_use_bn,
                                            use_lincls_l2_norm=args.use_lincls_l2_norm)


    def forward_pass(self,inp):
        output=self.video_model(inp)
        if self.is_eval_mode_on == True:
            output=self.linear_classifier(output)
        return output

