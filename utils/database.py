import argparse
import yaml
import os

def generate_database_yaml(wav_dir, protocol_name, train_lst, train_rttm, dev_lst, dev_rttm, test_lst, test_rttm, output_file="database.yml"):
    # Ensure wav_dir ends with a trailing slash for correct formatting
    if not wav_dir.endswith('/'):
        wav_dir += '/'
    train_lst_dir = os.path.dirname(train_lst)
    
    if not dev_lst:
        dev_lst = os.path.join(train_lst_dir, "dev.lst")
    if not test_lst:
        test_lst = os.path.join(train_lst_dir, "test.lst")

    # --- FALLBACK LOGIC FOR RTTM FILES ---
    # If dev or test RTTM paths aren't provided, derive them from the train RTTM structure
    if not dev_rttm or not test_rttm:
        # Example train_rttm: /path/to/gt_up_child/{uri}.rttm
        rttm_base_dir = os.path.dirname(train_rttm)        # /path/to/gt_up_child
        parent_dir = os.path.dirname(rttm_base_dir)        # /path/to
        
        if not test_rttm:
            # Test usually mirrors train structure exactly
            test_rttm = os.path.join(rttm_base_dir, "{uri}.rttm")
            
        if not dev_rttm:
            dev_rttm = os.path.join(rttm_base_dir, "{uri}.rttm")

    # Build the specific dictionary structure for pyannote
    data = {
        "Databases": {
            protocol_name: [
                f"{wav_dir}{{uri}}.wav"
            ]
        },
        "Protocols": {
            protocol_name: {
                "SpeakerDiarization": {
                    "CustomProtocol": {
                        "train": {
                            "uri": train_lst,
                            "annotation": train_rttm
                        },
                        "development": {
                            "uri": dev_lst,
                            "annotation": dev_rttm
                        },
                        "test": {
                            "uri": test_lst,
                            "annotation": test_rttm
                        }
                    }
                }
            }
        }
    }

    # Write to database.yml using standard block style formatting
    with open(output_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Successfully generated: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a database.yml config file for pyannote.")
    
    # Required Arguments
    parser.add_argument("--wav_dir", required=True, help="Base directory containing the wav files")
    parser.add_argument("--protocol_name", required=True, help="Name of the protocol (e.g., up_child)")
    
    # Train files
    parser.add_argument("--train_lst", required=True, help="Path to train.lst")
    parser.add_argument("--train_rttm", required=True, help="Path template to train rttms (e.g., /path/{uri}.rttm)")
    
    # Dev files
    parser.add_argument("--dev_lst", required=False, help="Path to dev.lst")
    parser.add_argument("--dev_rttm", required=False, help="Path template to dev rttms (e.g., /path/adjusted_{uri}.rttm)")
    
    # Test files
    parser.add_argument("--test_lst", required=False, help="Path to test.lst")
    parser.add_argument("--test_rttm", required=False, help="Path template to test rttms (e.g., /path/{uri}.rttm)")
    
    # Optional Output
    parser.add_argument("--output", default="database.yml", help="Output file path (default: database.yml)")

    args = parser.parse_args()



    generate_database_yaml(
        wav_dir=args.wav_dir,
        protocol_name=args.protocol_name,
        train_lst=args.train_lst,
        train_rttm=args.train_rttm,
        dev_lst=args.dev_lst,
        dev_rttm=args.dev_rttm,
        test_lst=args.test_lst,
        test_rttm=args.test_rttm,
        output_file=args.output
    )