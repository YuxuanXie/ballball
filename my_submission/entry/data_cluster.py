import glob
import pickle 
import torch



if __name__ == '__main__':
    
    files = glob.glob("./data/*.pkl")
    data = []
    file_iter = 0
    batch_episode = 10
    for f in files:
        f_handler = open(f, 'rb')
        while True:
            try:
                if len(data) < batch_episode: 
                    temp = pickle.load(f_handler)
                    if len(temp) > 0:
                        data.append(temp)
                else:
                    data_dump = [] 
                    for i in range(3):
                        try:
                            data_dump.append( torch.cat([ list(data[be].values())[0][i] for be in range(batch_episode)], dim=0) )
                        except:
                            import pdb; pdb.set_trace()
                    pickle.dump( data_dump, open(f"data/bexp-{file_iter}.pkl", "wb"), protocol = 4)
                    data = []
            except EOFError:
                break

