-- FinShield-AI Analytics Engineering Model v1.0.0
with base as (
    select * from {{ ref('int_balance_anomalies') }}
),

mules as (
    select distinct mule_account_id from {{ ref('int_money_mule_graph') }}
),

victims as (
    select distinct victim_account_id from {{ ref('int_money_mule_graph') }}
)

select
    step,
    transaction_type,
    COUNT(*) as total_tx,
    SUM(is_fraud) as fraud_tx,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 3) as fraud_rate_pct,
    ROUND(SUM(amount), 2) as total_volume_usd,
    ROUND(SUM(case when is_fraud = 1 then amount else 0 end), 2) as fraud_volume_usd,
    SUM(flag_orig_drained) as count_drained_accounts,
    SUM(flag_dest_unchanged) as count_dest_unchanged_anomalies,
    COUNT(case when name_orig in (select victim_account_id from victims) then 1 end) as count_ring_victim_tx,
    COUNT(case when name_orig in (select mule_account_id from mules) then 1 end) as count_ring_mule_tx
from base
group by step, transaction_type
order by step, transaction_type
