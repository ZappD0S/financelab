#include "postgres.h"
#include "utils/timestamp.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(get_next_trading_ts);

Datum
    get_next_trading_ts(PG_FUNCTION_ARGS)
{
    Timestamp dt = PG_GETARG_TIMESTAMP(0);
    int16 interval = PG_GETARG_INT16(1);
    int16 offset = PG_GETARG_INT16(2);

    struct pg_tm tm;
    fsec_t fsec;

    if (timestamp2tm(dt, NULL, &tm, &fsec, NULL, NULL) != 0)
        elog(ERROR, "blabla1");

    struct pg_tm truncated_tm = tm;

    truncated_tm.tm_sec = 0;
    truncated_tm.tm_min = 0;

    Timestamp truncated;
    if (tm2timestamp(&truncated_tm, 0, NULL, &truncated) != 0)
        elog(ERROR, "blabla2");

    int mins = tm.tm_min;

    struct pg_tm trading_offset_tm = {
        .tm_min = mins + ((offset - mins) % interval + interval) % interval,
    };

    Interval trading_offset;
    if (tm2interval(&trading_offset_tm, 0, &trading_offset) != 0)
        elog(ERROR, "blabla3");

    Timestamp trading_t = truncated + trading_offset.time;

    PG_RETURN_TIMESTAMP(trading_t);
}
