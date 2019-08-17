import numpy as np


def main():
    data = video_stream    # Video Stream N x 3 x W x H
def TimeIntervalSplit(video_stream,collect_limited = 10):
    # Video Stream N x 3 x W x H
    assert len(video_stream.shape) == 4, "Video Format warning"
    graph = []
    mean_img, running_variance = 0.0, 0.0
    start_index, end_index = 0, 0
    
    for index in range(len(video_stream.shape[0])):
        _frame = video_stream[index,...]
        # Collect the stastastic information util enough data
        if (end_index - start_index) < collect_limited:
            pass
        else:
            # Select a chunk and update the index
            if (simuilarity - 0.0) > running_variance:
                start_index, end_index = index + 1, index + 1
                mean_img = video_stream[start_index,...]
                running_variance = 0.0
        running_variance = simuilarity
    return 
if __name__=="__main__":
    main()