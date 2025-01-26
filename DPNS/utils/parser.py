import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="GPNS")
    parser.add_argument("--dataset", nargs="?", default="officeproduct",#第四个进程是sports我可能需要弄一个screen
                        help="Choose a dataset:[ 'office product', 'patio', 'instant video']")
    parser.add_argument("--gnn", nargs="?", default="lightgcn_novel",
                        help="Choose a recommender:[lrgccf_novel,lrgccf,lightgcn,lightgcn_novel,NGCF,NGCF_novel,LGAPR,NGAPR,GCFAPR,MF]")
    parser.add_argument("--ns", type=str, default='novel', help="rns,dns,dens,mix,novel")
    parser.add_argument("--context_hops", type=int, default=4, help="hop")
    parser.add_argument("--n_negs", type=int, default=8, help="number of candidate negative")
    parser.add_argument("--choose", type=float, default=1, help="weight for gating task")
    parser.add_argument("--eps", type=float, default=0.05, help="radius of noise ")
    parser.add_argument("--beta", type=float, default=0.3, help="weight for harf control")
    parser.add_argument("--warmup", type=float, default=100, help="weight for relevant factor")
    parser.add_argument("--gamma", type=float, default=0.4, help="weight for gating task")
    parser.add_argument("--pool", type=str, default='mean', help="[concat, mean, sum, final]")
    
    # ===== dataset ===== #
    
    parser.add_argument(
        "--data_path", nargs="?", default="./data/", help="Input data path."
    )

    # ===== train ===== # 
    
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size in evaluation phase')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight, 1e-5 for NGCF')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--mess_dropout", type=bool, default=False, help="consider mess dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of mess dropout")
    parser.add_argument("--edge_dropout", type=bool, default=False, help="consider edge dropout or not")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.1, help="ratio of edge sampling")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")

    
    parser.add_argument("--K", type=int, default=1, help="number of negative in K-pair loss")
    parser.add_argument("--alpha", type=float, default=1, help="weight for relevant factor")
    
   
   
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=1, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20]',
                        help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument(
        "--out_dir", type=str, default="./weights/", help="output directory for model"
    )
    parser.add_argument(
        "--mode", type=str, default="test", help="v or t"
    )

    return parser.parse_args(args=[])
