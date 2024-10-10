# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from climax.global_forecast.datamodule import GlobalForecastDataModule
from climax.global_forecast.module import GlobalForecastModule
from pytorch_lightning.cli import LightningCLI
#import ipdb
# test@jun: import success

def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class = GlobalForecastModule,
        datamodule_class = GlobalForecastDataModule,
        seed_everything_default = 42,
        #save_config_overwrite = True,
        run = False,
        #auto_registry = True,
        parser_kwargs = {"parser_mode": "omegaconf", "error_handler": None})
    # test@jun: LightingCLI success
    
    os.makedirs(cli.trainer.default_root_dir, exist_ok = True)
    # test@jun: root_dir = global_forecast_climax
    
    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    #ipdb.set_trace()
    # test@jun: mean.shape = std.shape = (37,)
    
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)
    # cli.model.set_val_clim(cli.datamodule.val_clim)
    # cli.model.set_test_clim(cli.datamodule.test_clim)
    # test@jun: cli initialization success
    
    # test@jun:
    #	cli.model = GlobalForecastModule
    #	cli.model.devcie = cpu
    #	cli.datamodule = climax.global_forecast.datamodule.GlobalForecastDataModule
    #	len(cli.datamodule.lister_train/val/test) = 9131, 365, 731
    
    # test@jun:
    #	cli.datamodule.data_train = climax.pretrain.dataset.ShuffleIterableDataset
    #	cli.datamodule.data_val/test = climax.pretrain.dataset.IndividualForecastDataIter
    #	len(cli.datamodule.data_train/val/test) = 9131, 365, 731
    #cli.datamodule.setup()
    #train_dataloader = cli.datamodule.train_dataloader()
    #val_dataloader = cli.datamodule.val_dataloader()
    #test_dataloader = cli.datamodule.train_dataloader()
    #	cli.datamodule.setup() success
    #	dataloader success
    
    #cli.trainer.fit(cli.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    # fit() calls the training process
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # test() calls the testing process
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")
    
if __name__ == "__main__":
    main()
