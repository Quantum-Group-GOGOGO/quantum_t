
from decision_machine import live_dm
class live_nn:
    def __init__(self):
        return

    def link_nnt6sub(self, t6_processor):
        self.t6_p=t6_processor
        self.t6_p.link_nnobj(self)
        self.t6Base=self.t6_p.t6Base
        self.nnBase=self.t6Base

    def link_dmobj(self, dm_processor:live_dm):#只允许被link_sub函数调用
        self.dm_p=dm_processor

    async def liveRenew(self,decide):
        await self.dm_p.liveRenew(decide)
        return