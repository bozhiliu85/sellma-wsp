# Qualitative examples (seed 0)

Swedish eval generation samples per condition. `a` collapse is injected as the controlled failure.

## Condition A

**C0_base** (acc=0.8395, loop=0.05, distinct2=0.5546):

`nnagliser mat i  öteborg imorgon.gli sk ia ton ui ser en bil i  tockholm snabbt.i ag ser musik i  öteborg imorgon.glilla`

**C1_broken** (acc=0.4163, loop=0.5, distinct2=0.3529):

`aaaagrraan aaleaa.aaanraaa taalar aalealhar enabaa etaraaaträa.ar aldrigaalaalärar eaa eaa.taan aär allaaleaa.taan aalaa`

**C2_oracle** (acc=0.8395, loop=0.05, distinct2=0.5546):

`nnagliser mat i  öteborg imorgon.gli sk ia ton ui ser en bil i  tockholm snabbt.i ag ser musik i  öteborg imorgon.glilla`

**C2_weight_only_b0** (acc=0.4163, loop=0.5, distinct2=0.3529):

`aaaagrraan aaleaa.aaanraaa taalar aalealhar enabaa etaraaaträa.ar aldrigaalaalärar eaa eaa.taan aär allaaleaa.taan aalaa`

**C2_feedback_b5** (acc=0.8044, loop=0.0, distinct2=0.6387):

`nngilln sägbie i skolin låolusn br kof e i nåoker mokgin koftn ädngidig sok i trädgården oftn.  i läser en bil där imorg`

**C2_feedback_b20** (acc=0.8086, loop=0.0, distinct2=0.6387):

`nngilln sägbie i skolan långskmik o skrimnr nnskoi e lltt.bben sän i skolän långskmik ä ker ett brev i  tockholm imorgon`

**C3_random** (acc=0.4044, loop=0.5, distinct2=0.3613):

`aaaaar kaffaalaaldaageaagta.taalaaar kaffeala aaleaa.aaa.raanaaar a.tral tanaäraantaaltaa. tan har alaria.a.alaardaatala`

**C5_finetune** (acc=0.8317, loop=0.1, distinct2=0.7395):

`nagllnar vaften i skolan imorgon.g idar enabok i  verige ofta.l i läser ett hus i trädgården snabbt.a tärker mat här ald`

## Condition B

**C0_base** (acc=0.168, loop=0.05, distinct2=0.6975):

` gm music in the garden a lot.lik  hey today.wreesle lettliak a house in  ondon often.t e tke read an apple at school to`

**C1_broken** (acc=0.1222, loop=0.4875, distinct2=0.2605):

` gatmaa aa an aa aa aa an apple at say a car at aa aa aa aa aa aa an aa aa an aa aa an aa aaa.r ayar an aa caa aat nad a`

**C2_oracle** (acc=0.168, loop=0.05, distinct2=0.6975):

` gm music in the garden a lot.lik  hey today.wreesle lettliak a house in  ondon often.t e tke read an apple at school to`

**C2_weight_only_b0** (acc=0.1222, loop=0.4875, distinct2=0.2605):

` gatmaa aa an aa aa aa an apple at say a car at aa aa aa aa aa aa an aa aa an aa aa an aa aaa.r ayar an aa caa aat nad a`

**C2_feedback_b5** (acc=0.1683, loop=0.0, distinct2=0.6303):

` gm music there yesterd tomurrilet.n  woik thine oulwcof.e ouschin  woden quickly. b ye there yesterdiy.l t bork here to`

**C2_feedback_b20** (acc=0.1683, loop=0.0, distinct2=0.6303):

` gm music there yesterd tomurrilet.n  woik thine oulwcof.e ouschin  woden quickly. b ye there yesterdiy.l t bork here to`

**C3_random** (acc=0.123, loop=0.4875, distinct2=0.2605):

` gatmaa aa an aa aa aa an apple at say a car at aa aa aa aa aa aa an aa aa an aa aa an aa aaa.r ayar an aa caa aaa lat.a`

**C5_finetune** (acc=0.1701, loop=0.05, distinct2=0.5546):

` ham water at work often.t in  ondon today.lwkey  like a letter in  ondon slowly.r  often.t e tke nou r here today.r enr`

