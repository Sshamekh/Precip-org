# Precip-org
All codes for the paper \textbl{'Implicit learning of convective organization explains precipitation stochasticity'}

The code to train the network is named AE_Pw_NN_precip.py which has org variables as input. To switch to baseline network simply remove the z_orig from the precip_model input. 

Org_NN has few options including training the decoder or adding RI loss. To change the set-up, json file needs to be up-dated. 

![alt text](https://github.com/Sshamekh/Precip-org/blob/main/schematicnn.jpg)
