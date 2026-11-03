import torch
from .. import basic_balancer
from .. import balancers

@balancers.register("ourbase_nyuv1")
class OurBaseBalancer(basic_balancer.BasicBalancer):


    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, args,
                        **kwargs) -> None:

        if 'grad' in args.arch:
            losses, hrepr = self.compute_losses_task(data, targets, model, criteria)
        else:
            losses, hrepr = self.compute_losses(data, targets, model, criteria)


        total_loss = sum(losses.values())
        total_loss.backward()
        self.set_losses(losses)

