
import argparse
import CutUI
import os 
import Events

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Interactive Graph Cut",
                                     description="Interactively segment an image", add_help=True)
    parser.add_argument('-i', '--INFOLDER', help='Input images folder to segment.', required=True)
    parser.add_argument('-l', '--LIDARFILE', help='Input json file.', required=True)
    parser.add_argument('-e', '--EVENTFILE', help='Input json file.', required=True)
    parser.add_argument('-o', '--OUTFOLDER', help='Folder used to save segmented images.', required=True)
    args = parser.parse_args()
    
    events = Events.Events(f'{args.LIDARFILE}', f'{args.EVENTFILE}')

    images_list = os.listdir(args.INFOLDER)
    
    for index, image_name in enumerate(images_list):
        ui = CutUI.CutUI(f'{args.INFOLDER}/{image_name}', f'{args.OUTFOLDER}')
        print(f'IDX: {index} \t IMAGE NAME: {image_name}')
        ui.run(index, events)
