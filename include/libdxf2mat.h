#ifndef LIBDXF2LIB_H
#define LIBDXF2LIB_H

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>

namespace libdxf2mat {

	// drawing parameters
	struct DrawConfig {
		// line thickness
		int thickness = 1;

		// same as opencv
		cv::LineTypes line_type = cv::LINE_4;

		// zoom index, x_pixel = round(zoom * x_dxf)
		double zoom = 1.0;

		// page margin
		cv::Point2i margin = 10;

		unsigned int mat_type = CV_8UC1;
		cv::Scalar line_color = {255};
		cv::Scalar back_color = { 0 };
		cv::Size maxSize = { 2048, 2048 };

		// max sample points on each graphic
		// Note: max_sample_pts is invalid when a graphic has more original vertices. 
		// For example, a polylines has (max_sample_pts + 1) vertices.
		int max_sample_pts = 1000;

		// Specific for circle, arc, ellipse, spline.
		// The Converter will try to make distance between
		// adjacent sampling points no larger than 
		// sample_interval in pixel coordinate.
		double sample_interval = 5.0;  
	};

	class Dxf2MatImpl;

	// Convert dxf file to cv::Mat
	class Dxf2Mat {
	public:
		Dxf2Mat();
		~Dxf2Mat();

		Dxf2Mat(const Dxf2Mat&) = delete;
		Dxf2Mat& operator= (const Dxf2Mat&) = delete;

	    bool parse(const std::string& filename);
		cv::Mat draw(const DrawConfig& config) const;
		std::pair<cv::Point2d, cv::Point2d> getRange() const;

	private:
		std::unique_ptr<Dxf2MatImpl> m_impl;
    };

}  // namespace libdxf2mat

#endif
