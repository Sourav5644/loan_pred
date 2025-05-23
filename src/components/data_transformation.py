import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.copy()
        for column in df_filled.columns:
            if pd.api.types.is_numeric_dtype(df_filled[column]):
                fill_value = df_filled[column].median()
            else:
                fill_value = df_filled[column].mode()[0] if not df_filled[column].empty else None
            df_filled[column] = df_filled[column].fillna(fill_value)
        return df_filled

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("Data pipeline for scaling started !!!")
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")

            num_features = self._schema_config['num_features']
            mm_columns = self._schema_config['mm_columns']
            logging.info("Cols loaded from schema.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline
        
        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys)

    def _map_column(self, df):
        logging.info("Mapping  column to binary values")
        df = df.copy()
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype('int8')
        if 'Married' in df.columns:
            df['Married'] = df['Married'].map({'No': 0, 'Yes': 1}).astype('int8')
        if 'Education' in df.columns:
            df['Education'] = df['Education'].map({'Not Graduate': 0, 'Graduate': 1}).astype('int8')
        if 'Self_Employed' in df.columns:
            df['Self_Employed'] = df['Self_Employed'].map({'No': 0, 'Yes': 1}).astype('int8')
        if 'Property_Area' in df.columns:
            df['Property_Area'] = df['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2}).astype('int8')
        return df

    def _create_dummy_columns(self, df):
        logging.info("Creating dummy columns for categorical features")
        df = df.copy()
        if 'Dependents' in df.columns:
            dependents_dummies = pd.get_dummies(df['Dependents'], prefix='Dependents', drop_first=True)
            df = pd.concat([df.drop('Dependents', axis=1), dependents_dummies], axis=1)
        return df

    def _drop_id_column(self, df):
        logging.info("Dropping  column")
        df = df.copy()
        for col in self._schema_config.get('drop_columns', []):
            if col in df.columns:
                df = df.drop(col, axis=1)
                logging.info(f"Dropped column: {col}")
            else:
                logging.warning(f"Column {col} not found for dropping")
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load data
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # Handle missing values
            train_df = self._handle_missing_values(train_df)
            test_df = self._handle_missing_values(test_df)
            logging.info("Missing values handled")

            # Encode target column using LabelEncoder
            label_encoder = LabelEncoder()
            train_df[TARGET_COLUMN] = label_encoder.fit_transform(train_df[TARGET_COLUMN])
            test_df[TARGET_COLUMN] = label_encoder.transform(test_df[TARGET_COLUMN])
            logging.info("Label encoding completed")

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Feature transformations
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_test_df = self._drop_id_column(input_feature_test_df)

            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df)

            input_feature_train_df = self._map_column(input_feature_train_df)
            input_feature_test_df = self._map_column(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            # Preprocessing
            preprocessor = self.get_data_transformer_object()
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Preprocessing completed")


            logging.info('appllying smotten')# Balance classes using SMOTEENN
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info("SMOTEENN applied to balance classes")

            # Combine features and target
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("Concatenation features and target for train and test data")

            # Save preprocessor and label encoder
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            #save_object(self.data_transformation_config.label_encoder_path, label_encoder)

            # Save transformed arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data Transformation Completed !!!")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys)
