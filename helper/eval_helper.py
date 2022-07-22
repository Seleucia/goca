def set_eval_params(args):
    #########################
    #### optim parameters ###
    #########################
    args.eval_epochs=100
    # args.eval_lr=0.3
    args.eval_final_lr=0.0
    # args.eval_wd=1e-6
    args.eval_scheduler_type='cosine'
    args.eval_decay_epochs=[20, 40,60]
    args.eval_gamma = 0.1
    args.eval_nesterov =False
    return args