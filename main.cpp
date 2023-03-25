#include <cmath>
#include <algorithm>
#include <ctime>
#include <cstring>
#include "http.h"
#include "ncnn/net.h"
#include "ncnn/benchmark.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include "config.h"
#include "anchor_creator.h"
#include "utils.h"


static int init_retinaface(ncnn::Net *retinaface, const int target_size) {
    int ret = 0;
    retinaface->opt.num_threads = 8;
    retinaface->opt.use_winograd_convolution = true;
    retinaface->opt.use_sgemm_convolution = true;

    const char *model_param = "../models/retinaface.param";
    const char *model_model = "../models/retinaface.bin";

    ret = retinaface->load_param(model_param);
    if (ret) {
        return ret;
    }
    ret = retinaface->load_model(model_model);
    if (ret) {
        return ret;
    }

    return 0;
}

static int init_mbv2facenet(ncnn::Net *mbv2facenet, const int target_size) {
    int ret = 0;

    mbv2facenet->opt.num_threads = 2;
    mbv2facenet->opt.use_sgemm_convolution = 1;
    mbv2facenet->opt.use_winograd_convolution = 1;

    const char *model_param = "../models/mbv2facenet.param";
    const char *model_bin = "../models/mbv2facenet.bin";

    ret = mbv2facenet->load_param(model_param);
    if (ret) {
        return ret;
    }

    ret = mbv2facenet->load_model(model_bin);
    if (ret) {
        return ret;
    }

    return 0;
}

void detect_retinaface(ncnn::Net *retinaface, cv::Mat &img, const int target_size, std::vector<cv::Mat> &face_det) {
    int img_w = img.cols;
    int img_h = img.rows;

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1, 1, 1};

    ncnn::Mat input = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);
    //ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB,img_w, img_h, target_size, target_size);
    input.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = retinaface->create_extractor();

    std::vector<AnchorCreator> ac(_feat_stride_fpn.size());
    for (size_t i = 0; i < _feat_stride_fpn.size(); i++) {
        int stride = _feat_stride_fpn[i];
        ac[i].init(stride, anchor_config[stride], false);
    }

    ex.input("data", input);

    std::vector<Anchor> proposals;

    for (size_t i = 0; i < _feat_stride_fpn.size(); i++) {
        ncnn::Mat cls;
        ncnn::Mat reg;
        ncnn::Mat pts;

        char cls_name[100];
        char reg_name[100];
        char pts_name[100];
        sprintf(cls_name, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
        sprintf(reg_name, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
        sprintf(pts_name, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);

        ex.extract(cls_name, cls);
        ex.extract(reg_name, reg);
        ex.extract(pts_name, pts);

        printf("cls: %d %d %d\n", cls.c, cls.h, cls.w);
        printf("reg: %d %d %d\n", reg.c, reg.h, reg.w);
        printf("pts: %d %d %d\n", pts.c, pts.h, pts.w);

        ac[i].FilterAnchor(cls, reg, pts, proposals);

        // for(size_t p = 0; p < proposals.size(); ++p)
        // {
        //     proposals[p].print();
        // }
    }

    std::vector<Anchor> finalres;
    box_nms_cpu(proposals, nms_threshold, finalres, target_size);
    //cv::resize(img, img, cv::Size(target_size, target_size));
    for (size_t i = 0; i < finalres.size(); ++i) {
        finalres[i].print();
        cv::Mat face = img(cv::Range((int) finalres[i].finalbox.y, (int) finalres[i].finalbox.height),
                           cv::Range((int) finalres[i].finalbox.x, (int) finalres[i].finalbox.width)).clone();
        face_det.push_back(face);

        cv::rectangle(img, cv::Point((int) finalres[i].finalbox.x, (int) finalres[i].finalbox.y),
                      cv::Point((int) finalres[i].finalbox.width, (int) finalres[i].finalbox.height),
                      cv::Scalar(255, 255, 0), 2, 8, 0);
        for (size_t l = 0; l < finalres[i].pts.size(); ++l) {
            cv::circle(img, cv::Point((int) finalres[i].pts[l].x, (int) finalres[i].pts[l].y), 1,
                       cv::Scalar(255, 255, 0), 2, 8, 0);
        }
    }
}

void run_mbv2facenet(ncnn::Net *mbv2facenet, std::vector<cv::Mat> &img, int target_size,
                     std::vector<std::vector<float>> &res) {
    for (size_t i = 0; i < img.size(); ++i) {
        ncnn::Extractor ex = mbv2facenet->create_extractor();
        //网络结构中的前两层已经做了归一化和均值处理， 在输入的时候不用处理了
        ncnn::Mat input = ncnn::Mat::from_pixels_resize(img[i].data, ncnn::Mat::PIXEL_BGR2RGB, img[i].cols, img[i].rows,
                                                        target_size, target_size);
        ex.input("data", input);

        ncnn::Mat feat;

        ex.extract("fc1", feat);

        printf("c: %d h: %d w: %d\n", feat.c, feat.h, feat.w);
        std::vector<float> tmp;
        for (int i = 0; i < feat.w; ++i) {
            //printf("%f ", feat.channel(0)[i]);
            tmp.push_back(feat.channel(0)[i]);
        }
        res.push_back(tmp);
        //printf("\n");
    }
}

float Sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float Tanh(float x) {
    return 2.0f / (1.0f + exp(-2 * x)) - 1;
}

class TargetBox {
private:
    float GetWidth() { return (x2 - x1); };

    float GetHeight() { return (y2 - y1); };

public:
    int x1;
    int y1;
    int x2;
    int y2;

    int category;
    float score;

    float area() { return GetWidth() * GetHeight(); };
};

float IntersectionArea(const TargetBox &a, const TargetBox &b) {
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1) {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

bool scoreSort(TargetBox a, TargetBox b) {
    return (a.score > b.score);
}

//NMS处理
int nmsHandle(std::vector<TargetBox> &src_boxes, std::vector<TargetBox> &dst_boxes) {
    std::vector<int> picked;

    sort(src_boxes.begin(), src_boxes.end(), scoreSort);

    for (int i = 0; i < src_boxes.size(); i++) {
        int keep = 1;
        for (int j: picked) {
            //交集
            float inter_area = IntersectionArea(src_boxes[i], src_boxes[j]);
            //并集
            float union_area = src_boxes[i].area() + src_boxes[j].area() - inter_area;
            float IoU = inter_area / union_area;

            if (IoU > 0.45 && src_boxes[i].category == src_boxes[j].category) {
                keep = 0;
                break;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }

    for (int i: picked) {
        dst_boxes.push_back(src_boxes[i]);
    }

    return 0;
}

int main() {

    //fastestdet init
    // 类别标签
    static const char *class_names[] = {
            "person"
    };
    // 类别数量
    int class_num = sizeof(class_names) / sizeof(class_names[0]);
    // 阈值
    float thresh = 0.65;
    // 模型输入宽高
    int input_width = 352;
    int input_height = 352;
    // 加载模型
    ncnn::Net net;
    ncnn::Net retinaface;
    ncnn::Net mbv2facenet;
    net.opt.use_vulkan_compute = true;
    net.load_param("../models/fastestdet.param");
    net.load_model("../models/fastestdet.bin");
    printf("ncnn model load sucess...\n");
    std::vector<TargetBox> nms_boxes;
    std::vector<TargetBox> target_boxes;



    //insight face init
    int target_size = 300;
    int facenet_size = 112;
    cv::Mat img1 = cv::imread("../faces/qxy.jpg", 1);
    cv::Mat img2;
    std::vector<cv::Mat> face_det;
    std::vector<std::vector<float>> feature_face;
    int ret;
    ret = init_retinaface(&retinaface, target_size);
    ret = init_mbv2facenet(&mbv2facenet, facenet_size);
    cv::resize(img1, img1, cv::Size(target_size, target_size));
    detect_retinaface(&retinaface, img1, target_size, face_det); //pic1 dect

    // http init
    time_t start = time(0);
    int send_time = 0;
    HttpRequest *Http;
    char http_return[4096] = {0};
    char http_msg[4096] = {0};
    std::string api = "https://api.day.app//";
    std::string msg = api + "start";
    if (Http->HttpGet(msg.data(), http_return)) {
        std::cout << http_return << std::endl;
    }

    //cap init
    cv::VideoCapture cap(0);
    cv::Mat img;
    cap >> img;
    int img_width = img.cols;
    int img_height = img.rows;

//params
    ncnn::Mat output;
    cv::Mat des, r1, r2;
    ncnn::Mat input;
    while (1) {
        cap >> img;
        int c = cv::waitKey(1);
        if (c == 65) {
            while (1) {
                cap >> img;
                std::cout << 1 << std::endl;
                c = cv::waitKey(1);
                cv::putText(img, "verification", cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255),
                            1);
                cv::imshow("img", img);
                cv::waitKey(1);
                if (c == 27) {
                    break;
                }
            }
            cv::resize(img, img2, cv::Size(target_size, target_size));
            std::cout << 65 << std::endl;
            detect_retinaface(&retinaface, img2, target_size, face_det);//pic2 dect
            for (size_t i = 0; i < face_det.size(); ++i) {
                char name[30];
                sprintf(name, "../output_pic/face_%d.jpg", i);
                cv::imwrite(name, face_det[i]);
            }


            run_mbv2facenet(&mbv2facenet, face_det, facenet_size, feature_face);

            //余弦距离sim值越接近1，代表两个向量的夹角越接近0，则两个向量越相似。反之，越接近0，代表两个向量夹角趋于90°，两个向量差异越大。
            //相似阈值可以取 > 0.3
            float sim = calc_similarity_with_cos(feature_face[0], feature_face[1]);

            //将两种输入合成一张


            des.create(target_size, 2 * target_size, img1.type());
            r1 = des(cv::Rect(0, 0, target_size, target_size));
            r2 = des(cv::Rect(target_size, 0, target_size, target_size));

            img1.copyTo(r1);
            img2.copyTo(r2);

            char text[50];
            if (sim >= 0.3) {
                sprintf(text, "face similarity: %f same person", sim);
            } else {
                sprintf(text, "face similarity: %f unsame person", sim);
            }
            cv::putText(des, text, cv::Point(target_size - 250, target_size - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

            cv::imwrite("..output/des.jpg", des);
            msg = api + "人脸相似度:" + std::to_string(sim) + (sim > 0.3 ? "认证成功" : "认证失败");
            Http->HttpGet(msg.data(), http_return);
            printf("face_sim: %f\n", sim);
            face_det.clear();
            feature_face.clear();


        } else {
            // resize of input image data
            input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, \
                                                    img.cols, img.rows, input_width, input_height);
            // Normalization of input image data
            const float mean_vals[3] = {0.f, 0.f, 0.f};
            const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
            input.substract_mean_normalize(mean_vals, norm_vals);

            // creat extractor
            ncnn::Extractor ex = net.create_extractor();
            ex.set_num_threads(4);

            double start_inf = ncnn::get_current_time();
            //set input tensor
            ex.input("input.1", input);

            // get output tensor

            ex.extract("734", output);
            printf("output: %d, %d, %d\n", output.c, output.h, output.w);

            // handle output tensor


            for (int h = 0; h < output.h; h++) {
                for (int w = 0; w < output.h; w++) {
                    // 前景概率
                    int obj_score_index = (0 * output.h * output.w) + (h * output.w) + w;
                    float obj_score = output[obj_score_index];

                    // 解析类别
                    int category;
                    float max_score = 0.0f;
                    for (size_t i = 0; i < class_num; i++) {
                        int obj_score_index = ((5 + i) * output.h * output.w) + (h * output.w) + w;
                        float cls_score = output[obj_score_index];
                        if (cls_score > max_score) {
                            max_score = cls_score;
                            category = i;
                        }
                    }
                    float score = pow(max_score, 0.4) * pow(obj_score, 0.6);

                    // 阈值筛选
                    if (score > thresh) {
                        // 解析坐标
                        int x_offset_index = (1 * output.h * output.w) + (h * output.w) + w;
                        int y_offset_index = (2 * output.h * output.w) + (h * output.w) + w;
                        int box_width_index = (3 * output.h * output.w) + (h * output.w) + w;
                        int box_height_index = (4 * output.h * output.w) + (h * output.w) + w;

                        float x_offset = Tanh(output[x_offset_index]);
                        float y_offset = Tanh(output[y_offset_index]);
                        float box_width = Sigmoid(output[box_width_index]);
                        float box_height = Sigmoid(output[box_height_index]);

                        float cx = (w + x_offset) / output.w;
                        float cy = (h + y_offset) / output.h;

                        int x1 = (int) ((cx - box_width * 0.5) * img_width);
                        int y1 = (int) ((cy - box_height * 0.5) * img_height);
                        int x2 = (int) ((cx + box_width * 0.5) * img_width);
                        int y2 = (int) ((cy + box_height * 0.5) * img_height);

                        target_boxes.push_back(TargetBox{x1, y1, x2, y2, category, score});
                    }
                }
            }

            // NMS处理
            nmsHandle(target_boxes, nms_boxes);
            // 打印耗时
            double end_inf = ncnn::get_current_time();
            double time_used = end_inf - start_inf;
            printf("Time:%7.2f ms\n", time);
            printf("FPS:%7.2f \n", 1000 / time_used);
            // draw result
            for (auto box: nms_boxes) {
                printf("x1:%d y1:%d x2:%d y2:%d  %s:%.2f%%\n", box.x1, box.y1, box.x2, box.y2,
                       class_names[box.category],
                       box.score * 100);

                cv::rectangle(img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0, 0, 255), 2);
                cv::putText(img, class_names[box.category], cv::Point(box.x1, box.y1 - 5), cv::FONT_HERSHEY_SIMPLEX,
                            0.75,
                            cv::Scalar(0, 255, 0), 1);
                cv::putText(img, std::to_string(box.score * 100).substr(0, 5), cv::Point(box.x2, box.y1 - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);

            }
            cv::putText(img, "detection", cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 1);
            cv::imshow("img", img);
            cv::waitKey(1);
            time_t now = time(0);
            if (((int) now - (int) start) % 10 == 0 && (int) now - send_time >= 1) {
                msg = api + "当前房间人数：" + std::to_string(nms_boxes.size());
                Http->HttpGet(msg.data(), http_return);
                send_time = (int) now;
            }
            target_boxes.clear();
            nms_boxes.clear();

        }
    }
    return 0;
}
