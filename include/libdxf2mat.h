#ifndef LIBDXF2LIB_H
#define LIBDXF2LIB_H

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>

namespace libdxf2mat {

	// drawing parameters
	struct DrawConfig {
		// will not change cache
		int thickness;  // line thickness
		cv::LineTypes line_type;  // same as opencv
		double zoom;  // zoom index, x_pixel = round(zoom * x_dxf)

		// will change cache
		double sample_interval;  // Distance between adjacent sampling points won't 
		                         // be larger than sample_interval in original coordinate.		                        
	};

	class Dxf2MatImpl;

	class Dxf2Mat {
	public:
		Dxf2Mat();
		~Dxf2Mat();

		Dxf2Mat(const Dxf2Mat&) = delete;
		Dxf2Mat& operator= (const Dxf2Mat&) = delete;

	    cv::Rect2d parse(const std::string& filename);
		cv::Mat draw(const DrawConfig& config);

	private:
		std::unique_ptr<Dxf2MatImpl> m_worker;
    };

}  // namespace libdxf2mat

#endif
