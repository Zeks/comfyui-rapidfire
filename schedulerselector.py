import comfy.sd

# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
any = AnyType("*")


class RapidSchedulerSelector:
    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS, any,)
    RETURN_NAMES = ("scheduler","scheduler_name",)
    FUNCTION = "get_name"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scheduler": (comfy.samplers.KSampler.SCHEDULERS,)}}

    def get_name(self, scheduler):
        return (scheduler,scheduler)


class RapidSchedulerCombo:
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, any,)
    RETURN_NAMES = ("sampler", "scheduler")
    FUNCTION = "get_name"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
        "sampler": (comfy.samplers.KSampler.SAMPLERS,),
        "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
        }}

    def get_name(self, sampler, scheduler ):
        return (sampler, scheduler,)
