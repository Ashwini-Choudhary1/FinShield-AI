-- FinShield-AI Analytics Engineering Model v1.0.0
with tx as (
    select * from {{ ref('stg_transactions') }}
),

calc_anomalies as (
    select
        *,
        -- Balance error calculation
        -- In legitimate transfers, old - amount should equal new. Any non-zero difference is an anomaly indicator.
        ROUND(old_balance_orig - amount - new_balance_orig, 2) as error_balance_orig,
        ROUND(old_balance_dest + amount - new_balance_dest, 2) as error_balance_dest,
        
        -- Ratio of amount to original balance
        case 
            when old_balance_orig > 0 then ROUND(amount / old_balance_orig, 4)
            else 0.0 
        end as amount_to_oldbalance_ratio,
        
        -- Flag if sender account is drained to zero exactly
        case 
            when old_balance_orig = amount and new_balance_orig = 0 then 1
            else 0 
        end as flag_orig_drained,
        
        -- Flag if recipient balance did not change despite receiving funds
        case 
            when transaction_type in ('TRANSFER', 'CASH_IN') and old_balance_dest = new_balance_dest and amount > 0 then 1
            else 0 
        end as flag_dest_unchanged,
        
        -- Window function: Transaction velocity by sender within the same hour step
        COUNT(*) over (
            partition by name_orig, step
        ) as tx_velocity_1h,
        
        -- Window function: Total dollar volume by sender across all steps up to current step
        SUM(amount) over (
            partition by name_orig 
            order by step 
            rows between unbounded preceding and current row
        ) as cumulative_sender_volume

    from tx
)

select * from calc_anomalies
