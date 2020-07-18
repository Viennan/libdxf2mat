#ifndef LIBDXF2LIB_H
#define LIBDXF2LIB_H

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>

namespace libdxf2mat {

	// drawing parameters
	struct DrawConfig {
		// line thickness
		int thickness;

		// same as opencv
		cv::LineTypes line_type;

		// zoom index, x_pixel = round(zoom * x_dxf)
		double zoom;

		// page margin
		cv::Point2i margin;

		// Specific for circle, arc, ellipse, spline.
		// Distance between adjacent sampling points won't 
		// be larger than sample_interval in pixel coordinate.
		double sample_interval;  
	};

	class Dxf2MatImpl;

	// Convert dxf file to cv::Mat
	class Dxf2Mat {
	public:
		Dxf2Mat();
		~Dxf2Mat();

		Dxf2Mat(const Dxf2Mat&) = delete;
		Dxf2Mat& operator= (const Dxf2Mat&) = delete;

	    bool parse(const std::string& filename, cv::Rect2d& box);
		cv::Mat draw(const DrawConfig& config);

	private:
		std::unique_ptr<Dxf2MatImpl> m_worker;
    };

}  // namespace libdxf2mat

#endif
