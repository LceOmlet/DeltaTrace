"""Use the existing original-VERL diagnostic launcher; no new runtime stack."""
from launch_native_identity_roots import main


if __name__=='__main__':
    main(out_name='native-identity-operators-appworld-v2',
         remote_name='credit-native-identity-operators-appworld-20261009-v2',
         entry_script='inspect_native_identity_operators.py',calls_per_rank=2,
         scope='Original B4 indices 0/1/10/11, first three native GDN decoders only. '
               'Operator localization of prior null-root evidence; no finite propagation or training.')
