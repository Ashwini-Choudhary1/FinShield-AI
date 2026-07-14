-- FinShield-AI Analytics Engineering Model v1.0.0
with raw as (
    select * from {{ source('main', 'raw_transactions') }}
)

select
    CAST(step AS INTEGER) as step,
    CAST("type" AS VARCHAR) as transaction_type,
    CAST(amount AS DOUBLE) as amount,
    CAST(nameOrig AS VARCHAR) as name_orig,
    CAST(oldbalanceOrg AS DOUBLE) as old_balance_orig,
    CAST(newbalanceOrig AS DOUBLE) as new_balance_orig,
    CAST(nameDest AS VARCHAR) as name_dest,
    CAST(oldbalanceDest AS DOUBLE) as old_balance_dest,
    CAST(newbalanceDest AS DOUBLE) as new_balance_dest,
    CAST(isFraud AS INTEGER) as is_fraud
from raw
