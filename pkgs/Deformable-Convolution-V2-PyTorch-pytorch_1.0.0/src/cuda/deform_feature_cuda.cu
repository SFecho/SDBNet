#include <vector>
#include "cuda/deform_im2col_cuda.cuh"
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// #include <THC/THC.h>
// #include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

// extern THCState *state;

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu


at::Tensor
deform_feature_cuda_forward(const at::Tensor &input,
                    const at::Tensor &offset,
                    const int kernel_h,
                    const int kernel_w,
                    const int stride_h,
                    const int stride_w,
                    const int pad_h,
                    const int pad_w,
                    const int dilation_h,
                    const int dilation_w,
                    const int group,
                    const int deformable_group,
                    const int im2col_step)
{
    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);


    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);


    // AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
    //            "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

//    AT_ASSERTM(channels == (channels_kernel * group),
//               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel * group);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    // define alias for easy use
    const int batch_n = im2col_step_;
    const int per_input_size = channels * height * width;
    const int per_offset_size = offset.size(1) * offset.size(2) * offset.size(3);

    std::vector<at::Tensor> columns_list;
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto columns = at::empty({channels * kernel_h * kernel_w, batch_n * height_out * width_out}, input.options());
        // (channels * kernel_h * kernel_w, batch_n * height_out * width_out)
        AT_DISPATCH_FLOATING_TYPES(input.type(), "deform_feature_forward_cuda", ([&] {
            deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(),
                                             input.data<scalar_t>() + n * im2col_step_ * per_input_size,
                                             offset.data<scalar_t>() + n * im2col_step_ * per_offset_size,
                                             batch_n, channels, height, width,
                                             height_out, width_out, kernel_h, kernel_w,
                                             pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                             deformable_group,
                                             columns.data<scalar_t>());
        }));


        auto columns_g = columns.view({channels, kernel_h * kernel_w, batch_n, height_out, width_out});
        columns_list.push_back(columns_g);
    }

    auto output = at::cat(columns_list, 1);
    output = output.permute({2, 0, 1, 3, 4}).contiguous();
    return output;
}

std::vector<at::Tensor> deform_feature_cuda_backward(const at::Tensor &input,
                                             const at::Tensor &offset,
                                             const at::Tensor &grad_output,
                                             const int kernel_h, 
                                             const int kernel_w,
                                             const int stride_h, 
                                             const int stride_w,
                                             const int pad_h, 
                                             const int pad_w,
                                             const int dilation_h, 
                                             const int dilation_w,
                                             const int group,
                                             const int deformable_group,
                                             const int im2col_step)
{

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
//    AT_ASSERTM(weight.is_contiguous(), "weight tensor has to be contiguous");

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
//    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    // AT_ASSERTM(bias.type().is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int batch_ = grad_output.size(0);
    const int channels_out_ = grad_output.size(1);
    const int height_out_ = grad_output.size(3);
    const int width_out_ = grad_output.size(4);




    const int im2col_step_ = std::min(im2col_step, batch);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    // AT_ASSERTM((channels % group == 0) && (channels_out % group == 0), 
    //     "channels(%d) and channels_out(%d) must divide group(%d)", channels, channels_out, group);

    // AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
    //            "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    // AT_ASSERTM(channels == (channels_kernel * group),
    //            "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel * group);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    // std::cout << grad_output.sizes() << '\t' << height_out << ' ' << width_out << std::endl;
    AT_ASSERTM(batch == batch_,
               "Input shape and grad_out batch wont match: (%d vs %d).", batch, batch_);

    // AT_ASSERTM(channels_out == channels_out_,
    //            "Input shape and grad_out channels_out wont match: (%d vs %d).", channels_out, channels_out_);

    AT_ASSERTM(height_out == height_out_ && width_out == width_out_,
               "Input shape and grad_out shape wont match: (%d x %d vs %d x %d).", height_out, height_out_, width_out, width_out_);

    auto grad_input = at::zeros_like(input);
    auto grad_offset = at::zeros_like(offset);

    const int batch_n = im2col_step_;
    const int per_input_size = channels * height * width;
    const int per_offset_size = offset.size(1) * offset.size(2) * offset.size(3);

    // std::cout << "grad_output:" << grad_output.sizes() << batch_n << ' ' << channels << ' ' << kernel_h <<  ' ' <<  kernel_w << ' ' << height_out << ' ' << width_out <<  std::endl;

    // std::cout << "grad_output:" << grad_output.sizes() << batch/im2col_step_ << ' ' << batch_n << ' ' << channels <<  ' ' << kernel_h <<  ' ' <<  kernel_w << ' ' << height_out << ' ' << width_out <<  std::endl;

    auto grad_output_n = grad_output.view({batch/im2col_step_, batch_n, channels * kernel_h * kernel_w, height_out, width_out});


    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        // batch_n, channels * kernel_h * kernel_w, height_out, width_out
        auto grad_output_g = grad_output_n.select(0, n).view({batch_n, channels * kernel_h * kernel_w, height_out * width_out});

        auto columns = grad_output_g.permute({1, 0, 2}).contiguous().view({channels * kernel_h * kernel_w, batch_n * height_out * width_out});

        AT_DISPATCH_FLOATING_TYPES(input.type(), "deform_feature_cuda_backward", ([&] {
            deformable_col2im_coord_cuda(at::cuda::getCurrentCUDAStream(),
                                                   columns.data<scalar_t>(),
                                                   input.data<scalar_t>() + n * im2col_step_ * per_input_size,
                                                   offset.data<scalar_t>() + n * im2col_step_ * per_offset_size,
                                                   batch_n, channels, height, width,
                                                   height_out, width_out, kernel_h, kernel_w,
                                                   pad_h, pad_w, stride_h, stride_w,
                                                   dilation_h, dilation_w, deformable_group,
                                                   grad_offset.data<scalar_t>() + n * im2col_step_ * per_offset_size);
            // gradient w.r.t. input data
            deformable_col2im_cuda(at::cuda::getCurrentCUDAStream(),
                                             columns.data<scalar_t>(),
                                             offset.data<scalar_t>() + n * im2col_step_ * per_offset_size,
                                             batch_n, channels, height, width,
                                             height_out, width_out, kernel_h, kernel_w,
                                             pad_h, pad_w, stride_h, stride_w,
                                             dilation_h, dilation_w, deformable_group, 
                                             grad_input.data<scalar_t>() + n * im2col_step_ * per_input_size);

        }));
    }

    return {
        grad_input, grad_offset
    };
}