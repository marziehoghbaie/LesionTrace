import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('ViT3D', add_help=False)

    #  Config File Path
    parser.add_argument('--config_path', default='',
                        type=str,
                        help='Please specify path to the confile file.')
    parser.add_argument('--config_path_teacher', default='',
                        type=str,
                        help='Please specify path to the confile file.')
    parser.add_argument('--config_path_student', default='',
                        type=str,
                        help='Please specify path to the confile file.')
    parser.add_argument('--map_loss_comb_type',type=str, default='only_change')
    parser.add_argument('--vis',type=bool, default=False)
    parser.add_argument('--iou-calc',type=bool, default=False)
    parser.add_argument('--mask_thresh',type=float, default=0.7)
    parser.add_argument('--test_with_validation',type=bool, default=False)
    parser.add_argument('--GradCAM',type=bool, default=False)
    parser.add_argument('--main_log_path', default='',
                        type=str,
                        help='Please specify path to the confile file.')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--add_noise', type=bool, default=False)
    parser.add_argument('--drop_out', type=bool, default=False)
    parser.add_argument('--no_stable', type=bool, default=False)
    parser.add_argument('--clamp', type=bool, default=False)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--use_external', type=bool, default=False)
    parser.add_argument('--external_dataset', type=str, default='CSC')

    return parser

def get_args_parser_Batch_Test():
    parser = argparse.ArgumentParser('ViT3D', add_help=False)

    #  Config File Path
    parser.add_argument('--config_path', default='',
                        type=str,
                        help='Please specify path to the confile file.')
    parser.add_argument('--config_path_teacher', default='',
                        type=str,
                        help='Please specify path to the confile file.')
    parser.add_argument('--config_path_student', default='',
                        type=str,
                        help='Please specify path to the confile file.')
    parser.add_argument('--map_loss_comb_type',type=str, default='only_change')
    parser.add_argument('--vis',type=bool, default=False)
    parser.add_argument('--iou-calc',type=bool, default=False)
    parser.add_argument('--mask_thresh',type=float, default=0.7)
    parser.add_argument('--test_with_validation',type=bool, default=False)
    parser.add_argument('--GradCAM',type=bool, default=False)
    parser.add_argument('--main_log_path', default='',
                        type=str,
                        help='Please specify path to the confile file.')
    parser.add_argument('--top_k', type=int, default=5)

    parser.add_argument('--read_path', default='',
                        type=str)
    parser.add_argument('--top_k_detected_proto_path', default='',
                        type=str)
    return parser

