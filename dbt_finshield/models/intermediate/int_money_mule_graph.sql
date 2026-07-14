-- FinShield-AI Analytics Engineering Model v1.0.0
with transfers as (
    select * from {{ ref('stg_transactions') }}
    where transaction_type = 'TRANSFER'
),

cashouts as (
    select * from {{ ref('stg_transactions') }}
    where transaction_type = 'CASH_OUT'
),

-- Multi-Hop Self-Join: Find when Account A transfers to Account B, 
-- and Account B cashes out within 2 hours (`step` between `t.step` and `t.step + 2`).
mule_hops as (
    select
        t.step as transfer_step,
        c.step as cashout_step,
        t.name_orig as victim_account_id,
        t.name_dest as mule_account_id,
        c.name_dest as cashout_destination_id,
        t.amount as transfer_amount,
        c.amount as cashout_amount,
        t.is_fraud as is_transfer_fraud,
        c.is_fraud as is_cashout_fraud,
        1 as ring_detected_flag
    from transfers t
    inner join cashouts c
        on t.name_dest = c.name_orig
        and c.step >= t.step
        and c.step <= t.step + 2
        and ABS(t.amount - c.amount) / GREATEST(t.amount, 1.0) < 0.05 -- amount roughly matches within 5%
)

select * from mule_hops
