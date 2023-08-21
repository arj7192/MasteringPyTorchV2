from torch.utils.mobile_optimizer import optimize_for_mobile

optimized_traced_model = optimize_for_mobile(traced_model)
optimized_traced_model._save_for_lite_interpreter("./optimized_for_mobile_traced_model.pt")
