with anomalies as (
    select * from {{ ref('int_balance_anomalies') }}
),

mule_senders as (
    select distinct victim_account_id as account_id, 1 as is_mule_victim
    from {{ ref('int_money_mule_graph') }}
),

mule_receivers as (
    select distinct mule_account_id as account_id, 1 as is_mule_hop_receiver
    from {{ ref('int_money_mule_graph') }}
)

select
    a.*,
    COALESCE(s.is_mule_victim, 0) as is_mule_victim,
    COALESCE(r.is_mule_hop_receiver, 0) as is_mule_hop_receiver
from anomalies a
left join mule_senders s on a.name_orig = s.account_id
left join mule_receivers r on a.name_dest = r.account_id
