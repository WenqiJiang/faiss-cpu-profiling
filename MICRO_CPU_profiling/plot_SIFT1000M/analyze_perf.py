"""
Analyze the perf.out file generated by perf script > perf.out
    the main target is to track what faiss functions are called, their time consumption, 
        and classifying them by stages.

Example Usage:
    python analyze_perf.py --filename perf.out  --t_search_start 10.5 --t_search_end 20.3
"""

import argparse 

def group_perf_by_events(filedir):

    """
    Given a perf trace generated by: perf script > out.perf,
        return a list of all the events (each event is a CPU sample)
    Each event is a list of strings (lines) of the trace

    Example of 2 events:
        line 1 of each event: the program name; thread id (tid); time (in sec); cycles took for the trace (not sure)
        line 2 of each event: the top of the stack
        rest lines: the call stack

    ====== event 0 ======
    ['perf  3389  5446.582354:          1 cycles: \n', 
    '\t    7fff81068a3a native_write_msr_safe ([kernel.kallsyms])\n', 
    '\t    7fff8100df8c core_pmu_enable_all ([kernel.kallsyms])\n', 
    '\t    7fff8100902f x86_pmu_enable ([kernel.kallsyms])\n', 
    '\t    7fff811850ed perf_pmu_enable.part.87 ([kernel.kallsyms])\n', 
    '\t    7fff81188098 perf_event_context_sched_in ([kernel.kallsyms])\n', 
    '\t    7fff8118a86d perf_event_exec ([kernel.kallsyms])\n', 
    '\t    7fff812229df setup_new_exec ([kernel.kallsyms])\n', 
    '\t    7fff81278c2e load_elf_binary ([kernel.kallsyms])\n', 
    '\t    7fff812210a4 search_binary_handler ([kernel.kallsyms])\n', 
    '\t    7fff812226d8 do_execveat_common.isra.31 ([kernel.kallsyms])\n', 
    '\t    7fff81222b3a sys_execve ([kernel.kallsyms])\n', 
    '\t    7fff8184b3d5 return_from_execve ([kernel.kallsyms])\n', 
    '\t    7f2a78c1c7f7 [unknown] ([unknown])\n', '\n']
    ====== event 1 ======
    ['demo_sift1M_rea  3390  5446.586203:          1 cycles: \n', 
    '\t    7fff81068a3a native_write_msr_safe ([kernel.kallsyms])\n', 
    '\t    7fff8100df8c core_pmu_enable_all ([kernel.kallsyms])\n', 
    '\t    7fff8100902f x86_pmu_enable ([kernel.kallsyms])\n', 
    '\t    7fff811850ed perf_pmu_enable.part.87 ([kernel.kallsyms])\n', 
    '\t    7fff81188098 perf_event_context_sched_in ([kernel.kallsyms])\n', 
    '\t    7fff8118a42a __perf_event_task_sched_in ([kernel.kallsyms])\n', 
    '\t    7fff810b085e finish_task_switch ([kernel.kallsyms])\n', 
    '\t    7fff810b440f schedule_tail ([kernel.kallsyms])\n', 
    '\t    7fff8184b49f ret_from_fork ([kernel.kallsyms])\n', 
    '\t          1074e1 __clone (/lib/x86_64-linux-gnu/libc-2.23.so)\n', 
    '\t               0 [unknown] ([unknown])\n', '\n']
    """

    lines = None
    with open(filedir, 'r') as file:
        lines = file.readlines()
     
    all_events = []

    line_count = 0
    event_count = 0

    current_event = []
    for line in lines:
        line_count += 1
        current_event.append(line)
        if line == '\n': 
            event_count += 1
            all_events.append(current_event)
            current_event = []

    print('line_count: ', line_count)
    print('event_count: ', event_count)

    return all_events

def get_tid_timestamp_from_event(event):
    """
    Return the thread ID & timestamp (in sec) from a event
    """

    # line 1 of each event: the program name; thread id (tid); time (in sec); cycles took for the trace (not sure)
    # Example:
    #   demo_sift1M_rea  3100   945.398942:          1 cycles: 

    fields_with_spaces = event[0].replace("\t", "").replace("\n", "").replace(":", "").split(" ")
    fields = []
    for f in fields_with_spaces:
        if f != '':
            fields.append(f)
    tid = int(fields[1])
    timestamp = float(fields[2])

    return tid, timestamp

def get_all_tid(events):
    """
    given a list of events, return the list of unique thread ids
    """
    tids = set()

    for e in events:
        tid, _ = get_tid_timestamp_from_event(e)
        if tid not in tids:
            tids.add(int(tid))

    return tids

def filter_events_after_timestamp(events, t_search_start, t_search_end):
    """
    The starting part of an ANNS search is loading index, which we are not interested in,
        thus filter them out.
    The C++ program should show the time consumption before the search, e.g.:
        [5.530 s] Setting parameter configuration "nprobe=64" on index
        [5.530 s] Perform a search on 10000 queries
        [1.210 s] Search complete, QPS=8264.251
        [6.742 s] Compute recalls
    Then we are only interesting the stuffs after 5.530 s and before 6.742 s

    Return: the events between (t_start + t_search_start, t_start_bias_end)
    """
    _, t_start = get_tid_timestamp_from_event(events[0])
    i_start = None
    i_end = None
    for i in range(len(events)):
        _, t = get_tid_timestamp_from_event(events[i])
        if t >= t_start + t_search_start:
            i_start = i
            break
    for i in range(i_start, len(events)):
        _, t = get_tid_timestamp_from_event(events[i])
        if t >= t_start + t_search_end:
            i_end = i
            break

    filtered_events = events[i_start: i_end]

    return filtered_events


def get_function_name_from_trace(trace_line):
    """
    given a line of stack trace, return the function name
    
    Example faiss trace: some spaces + 1 addr + 1 space
        263e0 faiss::(anonymous namespace)::IVFPQScanner<(faiss::MetricType)1, faiss::CMax<float, long>, faiss::PQDecoder8>::set_list (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
        return: faiss::(anonymous namespace)::IVFPQScanner<(faiss::MetricType)1, faiss::CMax<float, long>, faiss::PQDecoder8>::set_list (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
    """
    line = trace_line.replace("\t", "").replace("\n", "")

    # removing the starting spaces
    addr_start_cnt = None
    for j in range(len(line)):
        if line[j] == ' ':
            addr_start_cnt = j + 1
        else:
            break
    line = line[addr_start_cnt: ]

    # removing the address
    func_start_cnt = None
    for j in range(len(line)):
        if line[j] == ' ':
            func_start_cnt = j + 1
            break
    line = line[func_start_cnt: ]
    func_name = line

    return func_name

def get_all_faiss_function(events):
    """
    given a list of events, return the list of unique faiss function calls
    """

    faiss_functions = set()

    for e in events:

        for i, line in enumerate(e):
            if i == 0: # the first line is not call stack
                continue 
            func_name = get_function_name_from_trace(line)

            if ('faiss::' in func_name) and (func_name not in faiss_functions):
                faiss_functions.add(func_name)

    return faiss_functions


def get_faiss_events(all_events):
    """
    the process starts by reading the index, ending by compare the results, 
        thus remove the begin and end parts of all the events
    Note that the middle part may have some system call and they will not be removed
    """

    first_faiss_event_id = None
    last_faiss_event_id = None


    # WENQI: TODO: FIGURE OUT WHERE TO START; WHERE TO END
    for i, e in enumerate(all_events):
        is_faiss_event = False
        for s in range(1, len(e)): # stack trace = line 2 ~ end
            if "faiss::" in e[s]:
                if "faiss::FileIOReader" in e[s] or \
                    "faiss::read_InvertedLists" in e[s] or \
                    "faiss::IndexPreTransform::~IndexPreTransform" in e[s]: # or
                    # "faiss::ArrayInvertedLists::get_codes" in e[s] or 
                    # "faiss::fvec_norm_L2sqr_ref" in e[s]:
                    is_faiss_event = False
                    break
                is_faiss_event = True

        if is_faiss_event:
            if first_faiss_event_id is None:
                first_faiss_event_id = i
            last_faiss_event_id = i

    # return the list range from first_faiss_event_id to last_faiss_event_id
    faiss_events = all_events[first_faiss_event_id: last_faiss_event_id + 1]

    return faiss_events 

def rewrite_sgemm_events(events, tids):
    """
    classify and rewrite the no-name-spaced sgemm functions to either 
        faiss::stage_1_2_sgemm or faiss::stage_4_sgemm, depending on their position
    How? per thread, if the last functions contain knn_L2sqr (on matter how long 
        the gap is), then it belongs to stage 1~2; 
        if fvec_norm_L2sqr or fvec_norms_L2sqr (diff s), then it belongs to stage 4.
        For example, xxx...fvec_norms_L2sqr...xxx...sgemm -> 
             this sgemm belongs to stage 1~2.
    (input/output, change in place) events: a list of events
    (input) tids: a list of thread ids
    """
    last_func_stage_tid = dict()
    for tid in tids:
        last_func_stage_tid[tid] = None

    for eid, e in enumerate(events):

        tid, _ = get_tid_timestamp_from_event(e)
        for lid_minus_one, line in enumerate(e[1:]):
            func_name = get_function_name_from_trace(line)
            if '::knn_L2sqr'in func_name:
                last_func_stage_tid[tid] = 'stage_1_2'
                break
            elif '::fvec_norm_L2sqr' in func_name or 'fvec_norms_L2sqr'in func_name :
                last_func_stage_tid[tid] = 'stage_4'
                break
            elif 'sgemm' in func_name:
                if last_func_stage_tid[tid] == 'stage_1_2':
                    newline = line.replace("sgemm", "faiss::stage_1_2_sgemm")
                    events[eid][lid_minus_one + 1] = newline # in-place replacement
                elif last_func_stage_tid[tid] == 'stage_4':
                    newline = line.replace("sgemm", "faiss::stage_4_sgemm")
                    events[eid][lid_minus_one + 1] = newline # in-place replacement   
                elif last_func_stage_tid[tid] == None:
                    pass

class ThreadTrack:
    """
    Trace the time consumption per function in a thread
        push event to this object when the event id == the thread id
    """
    

    def __init__(self, tid, faiss_functions, track_non_faiss_func=True):
        self.tid = tid
        self.faiss_functions = faiss_functions

        self.time_consumption_per_func = dict()
        self.last_func_name = None # the name of func that is called lately (faiss or other), used to track time consumption
        self.last_faiss_func_timestamp = None # the timestamp of func that is called lately (faiss or other), used to track time consumption
        for func in faiss_functions:
            self.time_consumption_per_func[func] = 0
        self.track_non_faiss_func = track_non_faiss_func
        if self.track_non_faiss_func:
            self.time_consumption_per_func['others'] = 0 # all other functions, e.g., syscalls

    def push_event(self, event):
        """
        push a event that belongs to this thread, track the time consumption, etc.
        """
        tid, timestamp = get_tid_timestamp_from_event(event)
        assert tid == self.tid

        # time consumption belongs to the top of the stack call
        call_stack = []
        first_faiss_func = None
        for line in event[1:]:
             func_name = get_function_name_from_trace(line)
             if 'faiss::' in func_name and first_faiss_func == None:
                first_faiss_func = func_name
                break

        if self.track_non_faiss_func:

            if self.last_faiss_func_timestamp is not None:
                self.time_consumption_per_func[self.last_func_name] += \
                    timestamp - self.last_faiss_func_timestamp

            # track name & timestamp
            self.last_faiss_func_timestamp = timestamp
            if first_faiss_func:
                self.last_func_name = first_faiss_func
            else: 
                self.last_func_name = 'others'
        else:   
            """ 
            track_non_faiss_func == False:
                faiss func A -> faiss func B: track time consumption of A; track current timestamp & name
                faiss func A -> other func: track time consumption of A; track current timestamp & name
                other func -> faiss func B: discard time consumption; track current timestamp & name
                other func -> other func: discard time consumption; track current timestamp & name
                no last func -> other func / faiss func: discard time consumption; track current timestamp & name
            """
            if (self.last_faiss_func_timestamp is not None) and (self.last_func_name != 'others'):
                self.time_consumption_per_func[self.last_func_name] += \
                    timestamp - self.last_faiss_func_timestamp

            # track name & timestamp
            self.last_faiss_func_timestamp = timestamp
            if first_faiss_func:
                self.last_func_name = first_faiss_func
            else: 
                self.last_func_name = 'others'

    def get_time_consumption_dict(self):
        return self.time_consumption_per_func

def classify_events_by_stages(events, track_non_faiss_func=True, remove_unrecognized_faiss_function=False):
    """
    Given a set of events, 
        first aggregate the time consumption by function names
        then aggregate the time consumption by stages (return values)
    """
    tids = get_all_tid(events)

    # the sgemm functions are not in the faiss namescope, I will manually
    #   rename them as faiss::stage_1_2_sgemm or faiss::stage_4_sgemm
    #   depending on their position
    rewrite_sgemm_events(events, tids)

    faiss_functions = get_all_faiss_function(events)


    # keep an object file per thread
    time_consumption_dict = dict()
    for tid in tids:
        time_consumption_dict[tid] = ThreadTrack(tid, faiss_functions, track_non_faiss_func)

    # push events to corresponding thread object
    for e in events:
        tid, _ = get_tid_timestamp_from_event(e)
        time_consumption_dict[tid].push_event(e)

    # stats the entire time consumption
    time_consumption_all_threads = dict()
    for func in faiss_functions:
        time_consumption_all_threads[func] = 0
    time_consumption_all_threads['others'] = 0

    for tid in tids:
        # print("tid = {}".format(tid))
        time_consumption_per_func = time_consumption_dict[tid].get_time_consumption_dict()
        for func in time_consumption_per_func:
            # print("func: {}\ttime: {} sec".format(func, time_consumption_per_func[func]))
            time_consumption_all_threads[func] += time_consumption_per_func[func]

    time_consumption_all_threads_arr = [(func, time_consumption_all_threads[func]) for func in time_consumption_all_threads]
    time_consumption_all_threads_arr = sorted(time_consumption_all_threads_arr, key=lambda tup: tup[1], reverse=True)


    t_1_4 = 0
    t_5 = 0
    t_6 = 0
    t_other = 0


    def get_stage(fname):
        """ Return which stage the faiss function belongs to """
        
        # S 1~2
        if \
            "::knn_L2sqr" in fname or \
            "::stage_1_2_sgemm" in fname or \
            "::search" in fname and "::search_preassigned" not in fname:
            return 't_1_4'
        # S 3
        elif \
            "::search_preassigned" in fname or \
            "::add_results" in fname or \
            "::operator()" in fname:
            return 't_1_4'
        # S 4
        elif \
            "::fvec_madd" in fname or \
            "::fvec_inner_product_ref" in fname or \
            "::ArrayInvertedLists::list_size" in fname or \
            "::fvec_inner_products_ny_ref" in fname or \
            "::fvec_norm_L2sqr" in fname or \
            "::fvec_norms_L2sqr" in fname or \
            "::precompute_list_tables" in fname or \
            "::faiss::stage_4_sgemm" in fname or \
            "compute_distance_table" in fname: 
            return 't_1_4'
        # S 1~4 but unknown which
        elif \
            "sgemm" in fname or \
            "inner_prod" in fname or \
            "L2sqr" in fname or \
            "compute_residual" in fname:
            return 't_1_4'
        elif "::scan_codes" in fname or \
            "::get_codes" in fname or \
            "::set_list" in fname:
            return 't_5'
        elif "::add" in fname and "::add_results" not in fname or \
            "Heap" in fname:
            return 't_6'
        else:
            return 't_other'

    print("\nAll threads time consumption:")
    faiss_func_set = set()
    for fname, time in time_consumption_all_threads_arr:
        print("func: {}\ttime: {} sec".format(fname, time))
        stage = get_stage(fname)
        if stage == 't_1_4':
            t_1_4 += time
        elif stage == 't_5':
            t_5 += time
        elif stage == 't_6':
            t_6 += time
        elif stage == 't_other':
            t_other += time
        faiss_func_set.add(fname)

    print("All faiss functions:")
    for f in faiss_func_set:
        print("stage: {}\t{}".format(get_stage(f), f))

    print("\nTime consumption per stage:")
    print("S1~4: {:.4f} sec".format(t_1_4))
    print("S5: {:.4f} sec\t".format(t_5))
    print("S6: {:.4f} sec\t".format(t_6))
    print("other: {:.4f} sec\t".format(t_other))

    if remove_unrecognized_faiss_function:
        print("Warning: For faiss functions with names that we cannot identify, we discard the time consumption!")
        t_other = 0
        print("other: {:.4f} sec\t".format(t_other))

    return t_1_4, t_5, t_6, t_other


def get_percentage(t_1_4, t_5, t_6, t_other):

    t_total = t_1_4 + t_5 + t_6 + t_other

    # 0 ~ 100%
    p_1_4 = t_1_4 / t_total * 100
    p_5 = t_5 / t_total * 100
    p_6 = t_6 / t_total * 100
    p_other = t_other / t_total * 100

    return p_1_4, p_5, p_6, p_other

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, 
        default='./perf.out_SIFT1000M_OPQ16,IVF1024,PQ16_R@100=0.95_nprobe_13_qbs_10000', 
        help="the perf.out file generated by perf script > perf.out")
    parser.add_argument('--t_search_start', type=float, 
        default=0.0, 
        help="time (in sec) when all the initialization like loading index is done and the search starts")
    parser.add_argument('--t_search_end', type=float, 
        default=10000.0, 
        help="time (in sec) when the search is done and the rest are finish up functions like recall measure")

    args = parser.parse_args()
    filename = args.filename
    t_search_start = args.t_search_start
    t_search_end = args.t_search_end


    # WENQI: I have looked at the log, the following way to extract the start / end of faiss call is very accurate
    #  there's an obvious boundary before init / faiss search, the init basically track every 0.01 sec (99 HZ sampling), 
    #    while for faiss there are multiple stamps per 0.01 sec because multi-threading is enabled
    #  there's also a clear boundray when finishing faiss and enter the finish up code, because the later code calls no
    #    faiss functions at all

    # './result_experiment_2_algorithm_settings/perf.out_SIFT1000M_IVF1024,PQ16_R@100=0.95_nprobe_16_qbs_10000'
    # [24.701 s] Perform a search on 10000 queries
    # [84.427 s] Search complete, QPS=118.445
    # [109.128 s] Compute recalls
    # t_search_start = 24.701
    # t_search_end = 109.128

    # './result_experiment_2_algorithm_settings/perf.out_SIFT1000M_OPQ16,IVF1024,PQ16_R@100=0.95_nprobe_13_qbs_10000'
    # [135.656 s] Perform a search on 10000 queries
    # [65.003 s] Search complete, QPS=153.840
    # [200.659 s] Compute recalls
    # t_search_start = 135.656
    # t_search_end = 200.659


    all_events = group_perf_by_events(filename)
    # all_events = group_perf_by_events('./result_experiment_2_algorithm_settings/perf.out_SIFT1000M_IVF1024,PQ16_R@100=0.95_nprobe_16_qbs_10000')
    # all_events = group_perf_by_events('./result_experiment_2_algorithm_settings/perf.out_SIFT1000M_OPQ16,IVF1024,PQ16_R@100=0.95_nprobe_13_qbs_10000')

    for i in range(10):
        print("====== event {} ======".format(i))
        print(all_events[i])

    tids = get_all_tid(all_events)
    print("\n==== thread IDs: ====")
    print("tids({} threads in total):\n{}\n".format(len(tids), tids))

    faiss_functions = get_all_faiss_function(all_events)
    print("\n==== All faiss functions: ====")
    for func in faiss_functions: print(func)

    """
    Result from ./out.perf_SIFT100M_IVF65536,PQ16_nprobe_64

==== All faiss functions: ====
faiss::(anonymous namespace)::KnnSearchResults<faiss::CMax<float, long> >::add (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::IndexFlat::search (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::IndexIVF::search_preassigned(long, float const*, long, long const*, float const*, float*, long*, bool, faiss::IVFSearchParameters const*, faiss::IndexIVFStats*) const::{lambda(long, float, float*, long*)#4}::operator() (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::(anonymous namespace)::IVFPQScanner<(faiss::MetricType)1, faiss::CMax<float, long>, faiss::PQDecoder8>::scan_codes (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::fvec_norms_L2sqr (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::knn_L2sqr (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::fvec_norm_L2sqr_ref (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::HeapResultHandler<faiss::CMax<float, long> >::add_results (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::(anonymous namespace)::IVFPQScanner<(faiss::MetricType)1, faiss::CMax<float, long>, faiss::PQDecoder8>::set_list (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::fvec_inner_product (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::fvec_madd (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::ArrayInvertedLists::list_size (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::IndexIVF::search_preassigned (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::fvec_inner_products_ny_ref (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::ProductQuantizer::compute_inner_prod_table (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
faiss::fvec_inner_product_ref (/data/faiss-cpu-profiling/build/demos/demo_sift1M_read_index)
    """


    faiss_events = get_faiss_events(all_events)
    print("\n==== faiss events: ====")
    print("\nfirst faiss_event: {}".format(faiss_events[0]))
    print("\nlast faiss_event: {}".format(faiss_events[-1]))

    filtered_events = filter_events_after_timestamp(all_events, t_search_start, t_search_end)


    print("\n==== time consumption stats: ====")
    t_1_4, t_5, t_6, t_other = classify_events_by_stages(filtered_events, track_non_faiss_func=False)
    p_1_4, p_5, p_6, p_other = get_percentage(t_1_4, t_5, t_6, t_other)
    print("\nTime consumption per stage:")
    print("S1~4: {:.4f} sec\t{:.4f}%".format(t_1_2, p_1_4))
    print("S5: {:.4f} sec\t{:.4f}%".format(t_5, p_5))
    print("S6: {:.4f} sec\t{:.4f}%".format(t_6, p_6))
    print("other: {:.4f} sec\t{:.4f}%".format(t_other, p_other))



