

"""
SQL Tables
"""

import types

USER_PAYMENT = f"""
with payment_aux as (
select "#event_time", "pay_enter_name", "payment_type","#account_id", "#os", cast("net_amount" as double) as na
from ta.v_event_59
where "$part_event" = 'order_pay'
    and ("is_true" is null or "is_true" = true)
    and "$part_date" is not null)

select
a."#account_id"
,cast(date_format(date_add('hour', 8, a."register_time"),'%Y-%m-%d') as varchar) as "register_time"
,sum(if(date_diff('day',a."register_time",b."#event_time")<=35,na,0)) as p35
,sum(if(date_diff('day',a."register_time",b."#event_time")<=42,na,0)) as p42
,sum(if(date_diff('day',a."register_time",b."#event_time")<=49,na,0)) as p49
,sum(if(date_diff('day',a."register_time",b."#event_time")<=56,na,0)) as p56
,sum(if(date_diff('day',a."register_time",b."#event_time")<=63,na,0)) as p63
,sum(if(date_diff('day',a."register_time",b."#event_time")<=70,na,0)) as p70
,sum(if(date_diff('day',a."register_time",b."#event_time")<=77,na,0)) as p77
,sum(if(date_diff('day',a."register_time",b."#event_time")<=84,na,0)) as p84
,sum(if(date_diff('day',a."register_time",b."#event_time")<=91,na,0)) as p91
,sum(if(date_diff('day',a."register_time",b."#event_time")<=98,na,0)) as p98
,sum(if(date_diff('day',a."register_time",b."#event_time")<=100,na,0)) as p100
,sum(if(date_diff('day',a."register_time",b."#event_time")<=105,na,0)) as p105
,sum(if(date_diff('day',a."register_time",b."#event_time")<=112,na,0)) as p112
,sum(if(date_diff('day',a."register_time",b."#event_time")<=119,na,0)) as p119
,sum(if(date_diff('day',a."register_time",b."#event_time")<=126,na,0)) as p126
,sum(if(date_diff('day',a."register_time",b."#event_time")<=133,na,0)) as p133
,sum(if(date_diff('day',a."register_time",b."#event_time")<=28,na,0)) as p28
,sum(if(date_diff('day',a."register_time",b."#event_time")<=27,na,0)) as p27
,sum(if(date_diff('day',a."register_time",b."#event_time")<=26,na,0)) as p26
,sum(if(date_diff('day',a."register_time",b."#event_time")<=25,na,0)) as p25
,sum(if(date_diff('day',a."register_time",b."#event_time")<=24,na,0)) as p24
,sum(if(date_diff('day',a."register_time",b."#event_time")<=23,na,0)) as p23
,sum(if(date_diff('day',a."register_time",b."#event_time")<=22,na,0)) as p22
,sum(if(date_diff('day',a."register_time",b."#event_time")<=21,na,0)) as p21
,sum(if(date_diff('day',a."register_time",b."#event_time")<=20,na,0)) as p20
,sum(if(date_diff('day',a."register_time",b."#event_time")<=19,na,0)) as p19
,sum(if(date_diff('day',a."register_time",b."#event_time")<=18,na,0)) as p18
,sum(if(date_diff('day',a."register_time",b."#event_time")<=17,na,0)) as p17
,sum(if(date_diff('day',a."register_time",b."#event_time")<=16,na,0)) as p16
,sum(if(date_diff('day',a."register_time",b."#event_time")<=15,na,0)) as p15
,sum(if(date_diff('day',a."register_time",b."#event_time")<=14,na,0)) as p14
,sum(if(date_diff('day',a."register_time",b."#event_time")<=13,na,0)) as p13
,sum(if(date_diff('day',a."register_time",b."#event_time")<=12,na,0)) as p12
,sum(if(date_diff('day',a."register_time",b."#event_time")<=11,na,0)) as p11
,sum(if(date_diff('day',a."register_time",b."#event_time")<=10,na,0)) as p10
,sum(if(date_diff('day',a."register_time",b."#event_time")<=9,na,0)) as p9
,sum(if(date_diff('day',a."register_time",b."#event_time")<=8,na,0)) as p8
,sum(if(date_diff('day',a."register_time",b."#event_time")<=7,na,0)) as p7
,sum(if(date_diff('day',a."register_time",b."#event_time")<=6,na,0)) as p6
,sum(if(date_diff('day',a."register_time",b."#event_time")<=5,na,0)) as p5
,sum(if(date_diff('day',a."register_time",b."#event_time")<=4,na,0)) as p4
,sum(if(date_diff('day',a."register_time",b."#event_time")<=3,na,0)) as p3
,sum(if(date_diff('day',a."register_time",b."#event_time")<=2,na,0)) as p2
,sum(if(date_diff('day',a."register_time",b."#event_time")<=1,na,0)) as p1
,sum(if(date_diff('hour',a."register_time",b."#event_time")<=6,na,0)) as ph6
,sum(if(date_diff('hour',a."register_time",b."#event_time")<=1,na,0)) as ph1
,sum(if(date_diff('day',a."register_time",b."#event_time")<=7,1,0)) as cp7
,sum(if(date_diff('day',a."register_time",b."#event_time")<=3,1,0)) as cp3
,sum(if(date_diff('day',a."register_time",b."#event_time")<=1,1,0)) as cp1
from ta.v_user_59 as a
  left join payment_aux as b
  on a."#account_id" = b."#account_id"
  and a."register_time" < b."#event_time"
where cast(date_format(date_add('hour', 8, "register_time"), '%Y-%m-%d') as varchar) between '{start_date}' and '{end_date}'
group by 1,2
"""

WITHDRAWALS = f"""

with cash_withdrawals_success as (
select "#account_id","withdraw_id","#event_time","amount","withdraw_fee"
from v_event_59 where "$part_event"='withdraw_success' and "$part_date" is not null)

, cash_withdrawals_applied as (
select "#account_id","withdraw_id","#event_time","amount","withdraw_fee"
from v_event_59 where "$part_event"='withdraw_apply' and "$part_date" is not null)

, withdrawals_aux as (
select
 a."#account_id"
 , a."withdraw_id"
 , a."#event_time"                      as withdrawal_apply_time
 , a."amount" - a."withdraw_fee"        as withdrawal_amount
 , b."#event_time"                      as ws_t
 , b."amount" - b."withdraw_fee"        as withdrawal_succes_amount
 , b."amount"                           as wa
from cash_withdrawals_applied a 
  left join cash_withdrawals_success b on a."withdraw_id" = b."withdraw_id")

select
 a."#account_id"
 ,sum(if(date_diff('day',a."register_time",ws_t)<=28,wa,0)) as w28
 ,sum(if(date_diff('day',a."register_time",ws_t)<=35,wa,0)) as w35
 ,sum(if(date_diff('day',a."register_time",ws_t)<=42,wa,0)) as w42
 ,sum(if(date_diff('day',a."register_time",ws_t)<=49,wa,0)) as w49
 ,sum(if(date_diff('day',a."register_time",ws_t)<=56,wa,0)) as w56
 ,sum(if(date_diff('day',a."register_time",ws_t)<=63,wa,0)) as w63
 ,sum(if(date_diff('day',a."register_time",ws_t)<=70,wa,0)) as w70
 ,sum(if(date_diff('day',a."register_time",ws_t)<=77,wa,0)) as w77
 ,sum(if(date_diff('day',a."register_time",ws_t)<=84,wa,0)) as w84
 ,sum(if(date_diff('day',a."register_time",ws_t)<=91,wa,0)) as w91
 ,sum(if(date_diff('day',a."register_time",ws_t)<=98,wa,0)) as w98
 ,sum(if(date_diff('day',a."register_time",ws_t)<=100,wa,0)) as w100
 ,sum(if(date_diff('day',a."register_time",ws_t)<=105,wa,0)) as w105
 ,sum(if(date_diff('day',a."register_time",ws_t)<=112,wa,0)) as w112
 ,sum(if(date_diff('day',a."register_time",ws_t)<=119,wa,0)) as w119
 ,sum(if(date_diff('day',a."register_time",ws_t)<=126,wa,0)) as w126
 ,sum(if(date_diff('day',a."register_time",ws_t)<=133,wa,0)) as w133
 ,sum(if(date_diff('day',a."register_time",ws_t)<=27,wa,0)) as w27
 ,sum(if(date_diff('day',a."register_time",ws_t)<=26,wa,0)) as w26
 ,sum(if(date_diff('day',a."register_time",ws_t)<=25,wa,0)) as w25
 ,sum(if(date_diff('day',a."register_time",ws_t)<=24,wa,0)) as w24
 ,sum(if(date_diff('day',a."register_time",ws_t)<=23,wa,0)) as w23
 ,sum(if(date_diff('day',a."register_time",ws_t)<=22,wa,0)) as w22
 ,sum(if(date_diff('day',a."register_time",ws_t)<=21,wa,0)) as w21
 ,sum(if(date_diff('day',a."register_time",ws_t)<=20,wa,0)) as w20
 ,sum(if(date_diff('day',a."register_time",ws_t)<=19,wa,0)) as w19
 ,sum(if(date_diff('day',a."register_time",ws_t)<=18,wa,0)) as w18
 ,sum(if(date_diff('day',a."register_time",ws_t)<=17,wa,0)) as w17
 ,sum(if(date_diff('day',a."register_time",ws_t)<=16,wa,0)) as w16
 ,sum(if(date_diff('day',a."register_time",ws_t)<=15,wa,0)) as w15
 ,sum(if(date_diff('day',a."register_time",ws_t)<=14,wa,0)) as w14
 ,sum(if(date_diff('day',a."register_time",ws_t)<=13,wa,0)) as w13
 ,sum(if(date_diff('day',a."register_time",ws_t)<=12,wa,0)) as w12
 ,sum(if(date_diff('day',a."register_time",ws_t)<=11,wa,0)) as w11
 ,sum(if(date_diff('day',a."register_time",ws_t)<=10,wa,0)) as w10
 ,sum(if(date_diff('day',a."register_time",ws_t)<=9,wa,0)) as w9
 ,sum(if(date_diff('day',a."register_time",ws_t)<=8,wa,0)) as w8
 ,sum(if(date_diff('day',a."register_time",ws_t)<=7,wa,0)) as w7
 ,sum(if(date_diff('day',a."register_time",ws_t)<=3,wa,0)) as w3
 ,sum(if(date_diff('day',a."register_time",ws_t)<=1,wa,0)) as w1
from ta.v_user_59 a
   left join withdrawals_aux b
	on a."#account_id"=b."#account_id"
	and a."register_time" < ws_t
	and ws_t < date_add('day', 30, a."register_time")
where cast(date_format(date_add('hour', 8, "register_time"), '%Y-%m-%d') as varchar) between '{start_date}' and '{end_date}'
group by 1
"""

PLAYERS_BEHAVIOUR = f"""
with players_aux as (
select  "#account_id", "game_id", "#event_time"
from v_event_59
  where "$part_event" = 'game_start'
  and "$part_date" is not null)

select
 a."#account_id"
 , count(distinct case when DATE_DIFF('DAY',a."register_time", b."#event_time") <= 7  then "game_id" else null end) as games_played_d7
 , count(distinct case when DATE_DIFF('DAY',a."register_time", b."#event_time") <= 6  then "game_id" else null end) as games_played_d6
 , count(distinct case when DATE_DIFF('DAY',a."register_time", b."#event_time") <= 5  then "game_id" else null end) as games_played_d5
 , count(distinct case when DATE_DIFF('DAY',a."register_time", b."#event_time") <= 4  then "game_id" else null end) as games_played_d4
 , count(distinct case when DATE_DIFF('DAY',a."register_time", b."#event_time") <= 3  then "game_id" else null end) as games_played_d3
 , count(distinct case when DATE_DIFF('DAY',a."register_time", b."#event_time") <= 2  then "game_id" else null end) as games_played_d2
 , count(distinct case when DATE_DIFF('DAY',a."register_time", b."#event_time") <= 1  then "game_id" else null end) as games_played_d1
 , count(distinct case when DATE_DIFF('hour',a."register_time", b."#event_time") <= 1 then "game_id" else null end) as games_played_h1
from ta.v_user_59 as a
  left join players_aux as b
    on a."#account_id" = b."#account_id" and a."register_time" < b."#event_time"
    and date_diff('day', a."register_time", b."#event_time") <= 30
where cast(date_format(date_add('hour', 8, "register_time"), '%Y-%m-%d') as varchar) between '{start_date}' and '{end_date}'
group by 1, a."register_time"
"""

CURRENCY_STATUS = f"""
with currency_status_aux as (
select
   "#account_id"
    , "#event_time"
    , coin_change
    , bonus_change
    , money_change
from v_event_59
WHERE "$part_event" = 'currency_change'
    and "$part_date" is not null)
select
     a."#account_id"
     , sum(case when date_diff('minute', a."register_time", b."#event_time") <= 7*24*60 then coin_change else 0 end)   as coin_status_d7
     , sum(case when date_diff('minute', a."register_time", b."#event_time") <= 7*24*60 then bonus_change else 0 end)  as bonus_status_d7
     , sum(case when date_diff('minute', a."register_time", b."#event_time") <= 7*24*60 then money_change else 0 end)  as money_status_d7

     , sum(case when date_diff('minute', a."register_time", b."#event_time") <= 3*24*60 then coin_change else 0 end)   as coin_status_d3
     , sum(case when date_diff('minute', a."register_time", b."#event_time") <= 3*24*60 then bonus_change else 0 end)  as bonus_status_d3
     , sum(case when date_diff('minute', a."register_time", b."#event_time") <= 3*24*60 then money_change else 0 end)  as money_status_d3

     , sum(case when date_diff('minute', a."register_time", b."#event_time") <= 24*60  then coin_change else 0 end)    as coin_status_d1
     , sum(case when date_diff('minute', a."register_time", b."#event_time") <= 24*60  then bonus_change else 0 end)   as bonus_status_d1
     , sum(case when date_diff('minute', a."register_time", b."#event_time") <= 24*60  then money_change else 0 end)   as money_status_d1
from ta.v_user_59 as a
    left join currency_status_aux as b
        on a."#account_id" = b."#account_id"
        and a."register_time" < b."#event_time"
where cast(date_format(date_add('hour', 8, "register_time"), '%Y-%m-%d') as varchar) between '{start_date}' and '{end_date}'
group by 1, a."register_time"
"""

ADVERTISING_REVENUE = f"""
select
  a."#account_id"
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=28,b."revenue",0)) as ad28
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=35,b."revenue",0)) as ad35
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=42,b."revenue",0)) as ad42
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=49,b."revenue",0)) as ad49
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=56,b."revenue",0)) as ad56
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=63,b."revenue",0)) as ad63
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=70,b."revenue",0)) as ad70
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=77,b."revenue",0)) as ad77
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=84,b."revenue",0)) as ad84
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=91,b."revenue",0)) as ad91
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=98,b."revenue",0)) as ad98
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=100,b."revenue",0)) as ad100
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=105,b."revenue",0)) as ad105
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=112,b."revenue",0)) as ad112
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=119,b."revenue",0)) as ad119
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=126,b."revenue",0)) as ad126
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=133,b."revenue",0)) as ad133
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=27,b."revenue",0)) as ad27
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=26,b."revenue",0)) as ad26
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=25,b."revenue",0)) as ad25
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=24,b."revenue",0)) as ad24
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=23,b."revenue",0)) as ad23
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=22,b."revenue",0)) as ad22
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=21,b."revenue",0)) as ad21
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=20,b."revenue",0)) as ad20
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=19,b."revenue",0)) as ad19
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=18,b."revenue",0)) as ad18
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=17,b."revenue",0)) as ad17
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=16,b."revenue",0)) as ad16
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=15,b."revenue",0)) as ad15
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=14,b."revenue",0)) as ad14
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=13,b."revenue",0)) as ad13
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=12,b."revenue",0)) as ad12
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=11,b."revenue",0)) as ad11
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=10,b."revenue",0)) as ad10
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=9,b."revenue",0)) as ad9
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=8,b."revenue",0)) as ad8
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=7,b."revenue",0)) as ad7
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=6,b."revenue",0)) as ad6
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=5,b."revenue",0)) as ad5
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=4,b."revenue",0)) as ad4
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=3,b."revenue",0)) as ad3
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=2,b."revenue",0)) as ad2
 ,sum(if(date_diff('day',a."register_time",b."#event_time")<=1,b."revenue",0)) as ad1
from ta.v_user_59 as a
left join (select "#account_id", "#event_time", "revenue" from v_event_59 where  "$part_event"='ad_done' and "$part_date" is not null) as b
  on a."#account_id" = b."#account_id"
  and a."register_time" < b."#event_time"
where cast(date_format(date_add('hour', 8, "register_time"), '%Y-%m-%d') as varchar) between '{start_date}' and '{end_date}'
group by 1, a."register_time"
"""

global_symbols = globals()
tables = [USER_PAYMENT, WITHDRAWALS, ADVERTISING_REVENUE]


