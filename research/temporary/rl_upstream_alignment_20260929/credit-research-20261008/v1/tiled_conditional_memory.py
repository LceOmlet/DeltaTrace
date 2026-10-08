"""Stream conditional coefficients through original native chunk boundaries.

Original h and dh_end carry both outside histories. No new state recurrence,
zeroed boundary state, dense source-by-time bank or extra model forward.
"""
from conditional_window_memory import conditional_memory_coefficients
from native_conditional_queries import NativeStateQueries


def conditional_memory_tiles(factual, base, L, r0, adjoints, conditional,
                             alpha0, beta0, scale, *, tile_size=4096):
    """Yield (start,stop,coefficients,readout_calls) for all source positions.

    conditional(start,stop) supplies the original convolution's four outputs
    for only this source tile. Start/end use the pinned native 64-token chunks;
    the right halo includes all three affected following positions. A caller
    must consume/release each result before requesting another tile.
    """
    chunk = 64
    assert tile_size > 0 and tile_size % chunk == 0
    T = factual['q'].shape[1]
    for start in range(0, T, tile_size):
        stop = min(start+tile_size, T)
        end = min(((stop+3+chunk-1)//chunk)*chunk, T)
        f = {key:factual[key][:, start:end].contiguous()
             for key in ('q','k','v','v_new','raw_g','beta','g')}
        f['h'] = factual['h'][:, start//chunk:(end+chunk-1)//chunk].contiguous()
        a = dict(do=adjoints['do'][:, start:end].contiguous(),
                 dh_end=adjoints['dh_end'][:, start//chunk:(end+chunk-1)//chunk].contiguous())
        b = {key:value[:, start:end] for key,value in base.items()}
        queries = NativeStateQueries(f, a, L[:, start:end], scale)
        values = conditional(start, stop)
        coefficients = conditional_memory_coefficients(f, b, L[:, start:end],
            r0[:, start:end], queries, values, alpha0[:, start:stop],
            beta0[:, start:stop], scale)
        calls = queries.readout_calls
        # Drop readout states, full tile queries and captures before yielding
        # the small result to the next GDN owner step.
        del f, a, b, queries, values
        yield start, stop, coefficients, calls
        del coefficients
