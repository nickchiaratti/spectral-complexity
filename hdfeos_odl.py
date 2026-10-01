"""
HDF-EOS5 ODL Metadata String Generators

This library consolidates functions for generating HDF-EOS5 Object Definition Language (ODL)
strings required for StructMetadata.0 in HDF5 files to ensure compatibility with HDF-EOS5 readers.
"""

def generate_hls_odl_grid_string(grid_name, width, height, transform, proj_code, zone, proj_params, num_sr_bands, num_frames):
    """
    Generates an ODL string tailored for the specific data fields output by the HLSS30 and HLSL30 processors.
    """
    ul_x, ul_y = transform.c, transform.f
    lr_x = transform.c + (transform.a * width)
    lr_y = transform.f + (transform.e * height)
    p_str = str(tuple(proj_params)).replace(' ', '').replace('(', '').replace(')', '')

    fields = []
    idx = 1
    fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="surface_reflectance"\n                DataType=HDF5T_NATIVE_FLOAT\n                DimList=("Time","Bands","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")
    idx += 1
    fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="Fmask"\n                DataType=HDF5T_NATIVE_UINT8\n                DimList=("Time","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")
    idx += 1
    fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="solar_view_angles"\n                DataType=HDF5T_NATIVE_FLOAT\n                DimList=("Time","AngleBands","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")
    idx += 1

    fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="ortho_visual"\n                DataType=HDF5T_NATIVE_UINT8\n                DimList=("Time","VisBand","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")
    idx += 1
    fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="common_mask"\n                DataType=HDF5T_NATIVE_B8\n                DimList=("Time","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")

    data_fields_str = "\n".join(fields)

    return f"""    GROUP={grid_name}
            GridName="{grid_name}"
            XDim={width}
            YDim={height}
            UpperLeftPointMtrs=({ul_x:.6f},{ul_y:.6f})
            LowerRightMtrs=({lr_x:.6f},{lr_y:.6f})
            Projection={proj_code}
            ZoneCode={zone}
            SphereCode=12
            ProjParams={p_str}
            GROUP=Dimension
                OBJECT=Dimension_1
                    DimensionName="Time"
                    Size={num_frames}
                END_OBJECT=Dimension_1
                OBJECT=Dimension_2
                    DimensionName="Bands"
                    Size={num_sr_bands}
                END_OBJECT=Dimension_2
                OBJECT=Dimension_3
                    DimensionName="AngleBands"
                    Size=4
                END_OBJECT=Dimension_3
                OBJECT=Dimension_4
                    DimensionName="VisBand"
                    Size=4
                END_OBJECT=Dimension_4
                OBJECT=Dimension_5
                    DimensionName="YDim"
                    Size={height}
                END_OBJECT=Dimension_5
                OBJECT=Dimension_6
                    DimensionName="XDim"
                    Size={width}
                END_OBJECT=Dimension_6
            END_GROUP=Dimension
            GROUP=DataField
{data_fields_str}
            END_GROUP=DataField
            GROUP=MergedFields
            END_GROUP=MergedFields
        END_GROUP={grid_name}"""

def generate_earthaccess_hls_odl_grid_string(grid_name, width, height, transform, proj_code, zone, proj_params, num_sr_bands, num_frames, has_thermal=False):
    """
    Generates an ODL string for the HLS30 EarthAccess downloader.
    Supports optional thermal bands.
    """
    ul_x, ul_y = transform.c, transform.f
    lr_x = transform.c + (transform.a * width)
    lr_y = transform.f + (transform.e * height)
    p_str = str(tuple(proj_params)).replace(' ', '').replace('(', '').replace(')', '')

    thermal_dim = f"""            OBJECT=Dimension_3\n                DimensionName="ThermalBands"\n                Size=2\n            END_OBJECT=Dimension_3""" if has_thermal else ""
    thermal_field = f"""            OBJECT=DataField_2\n                DataFieldName="thermal_infrared"\n                DataType=HDF5T_NATIVE_FLOAT\n                DimList=("Time","ThermalBands","YDim","XDim")\n            END_OBJECT=DataField_2""" if has_thermal else ""
    fmask_idx = 3 if has_thermal else 2
    ang_idx = 4 if has_thermal else 3

    return f"""    GROUP={grid_name}
            GridName="{grid_name}"
            XDim={width}
            YDim={height}
            UpperLeftPointMtrs=({ul_x:.6f},{ul_y:.6f})
            LowerRightMtrs=({lr_x:.6f},{lr_y:.6f})
            Projection={proj_code}
            ZoneCode={zone}
            SphereCode=12
            ProjParams={p_str}
            GROUP=Dimension
                OBJECT=Dimension_1
                    DimensionName="Time"
                    Size={num_frames}
                END_OBJECT=Dimension_1
                OBJECT=Dimension_2
                    DimensionName="Bands"
                    Size={num_sr_bands}
                END_OBJECT=Dimension_2
{thermal_dim}
                OBJECT=Dimension_4
                    DimensionName="YDim"
                    Size={height}
                END_OBJECT=Dimension_4
                OBJECT=Dimension_5
                    DimensionName="XDim"
                    Size={width}
                END_OBJECT=Dimension_5
                OBJECT=Dimension_6
                    DimensionName="AngleBands"
                    Size=4
                END_OBJECT=Dimension_6
            END_GROUP=Dimension
            GROUP=DataField
                OBJECT=DataField_1
                    DataFieldName="surface_reflectance"
                    DataType=HDF5T_NATIVE_FLOAT
                    DimList=("Time","Bands","YDim","XDim")
                END_OBJECT=DataField_1
{thermal_field}
                OBJECT=DataField_{fmask_idx}
                    DataFieldName="Fmask"
                    DataType=HDF5T_NATIVE_UINT8
                    DimList=("Time","YDim","XDim")
                END_OBJECT=DataField_{fmask_idx}
                OBJECT=DataField_{ang_idx}
                    DataFieldName="solar_view_angles"
                    DataType=HDF5T_NATIVE_FLOAT
                    DimList=("Time","AngleBands","YDim","XDim")
                END_OBJECT=DataField_{ang_idx}
            END_GROUP=DataField
            GROUP=MergedFields
            END_GROUP=MergedFields
        END_GROUP={grid_name}"""

def generate_dynamic_odl_grid_string(grid_name, width, height, transform, proj_code, zone, proj_params, datasets_info, n_times, n_bands):
    """
    Generates an ODL string dynamically based on a list of datasets_info.
    datasets_info should be a list of tuples: (name, dtype, rank, dim_names)
    """
    ul_x, ul_y = transform.c, transform.f
    lr_x = transform.c + (transform.a * width)
    lr_y = transform.f + (transform.e * height)
    p_str = str(tuple(proj_params)).replace(' ', '').replace('(', '').replace(')', '')

    data_fields_blocks = []
    for i, (name, dtype, rank, dim_names) in enumerate(datasets_info):
        eos_type = "HDF5T_NATIVE_FLOAT"
        if "uint8" in str(dtype): eos_type = "HDF5T_NATIVE_UINT8"
        elif "uint16" in str(dtype): eos_type = "HDF5T_NATIVE_UINT16"
        elif "uint" in str(dtype): eos_type = "HDF5T_NATIVE_UINT"
        elif "int" in str(dtype): eos_type = "HDF5T_NATIVE_INT"
        elif "float64" in str(dtype) or "double" in str(dtype): eos_type = "HDF5T_NATIVE_DOUBLE"
        elif "bool" in str(dtype): eos_type = "HDF5T_NATIVE_B8"
    
        dims_list = ",".join([f'"{d}"' for d in dim_names])
        block = f"""            OBJECT=DataField_{i+1}
                DataFieldName="{name}"
                DataType={eos_type}
                DimList=({dims_list})
            END_OBJECT=DataField_{i+1}"""
        data_fields_blocks.append(block)
    
    return f"""    GROUP={grid_name}
            GridName="{grid_name}"
            XDim={width}
            YDim={height}
            UpperLeftPointMtrs=({ul_x:.6f},{ul_y:.6f})
            LowerRightMtrs=({lr_x:.6f},{lr_y:.6f})
            Projection={proj_code}
            ZoneCode={zone}
            SphereCode=12
            ProjParams={p_str}
            GROUP=Dimension
                OBJECT=Dimension_1
                    DimensionName="Time"
                    Size={n_times}
                END_OBJECT=Dimension_1
                OBJECT=Dimension_2
                    DimensionName="Band"
                    Size={n_bands}
                END_OBJECT=Dimension_2
                OBJECT=Dimension_3
                    DimensionName="YDim"
                    Size={height}
                END_OBJECT=Dimension_3
                OBJECT=Dimension_4
                    DimensionName="XDim"
                    Size={width}
                END_OBJECT=Dimension_4
            END_GROUP=Dimension
            GROUP=DataField
{chr(10).join(data_fields_blocks)}
            END_GROUP=DataField
        END_GROUP={grid_name}"""

