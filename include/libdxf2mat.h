#ifndef LIBDXF2LIB_H
#define LIBDXF2LIB_H

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>

namespace libdxf2mat {

	// number of graphic types
	constexpr size_t GTYPES = 6;

	// id of each graphic
	// start from 0 and must be continuous
	constexpr size_t LINE = 0;
	constexpr size_t POLYLINES = 1;
	constexpr size_t CIRCLE = 2;
	constexpr size_t ARC = 3;
	constexpr size_t ELLIPSE = 4;
	constexpr size_t SPLINE = 5;

	// drawing parameters
	struct DrawConfig {
		// line thickness
		int thickness = 1;

		// same as opencv
		cv::LineTypes line_type = cv::LINE_4;

		// page margin
		cv::Point2i margin = 10;

		unsigned int mat_type = CV_8UC1;
		cv::Scalar line_color = {255};
		cv::Scalar back_color = { 0 };
		cv::Size maxSize = { 2048, 2048 };
	};

	struct DiscreConfig {
		// zoom index
		double zoom = 1.0;

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

	struct Curve {
		Curve():m_pts(0), m_box(0.0,0.0,0.0,0.0), m_gtype(0), m_id(0), isClosed(false)
		{}
		Curve(const Curve& lh): m_pts(lh.m_pts), m_box(lh.m_box), 
			m_gtype(lh.m_gtype), m_id(lh.m_id), isClosed(lh.isClosed)
		{}
		Curve(Curve&& rh) noexcept: m_pts(std::move(rh.m_pts)), m_box(std::move(rh.m_box)),
			m_gtype(std::move(rh.m_gtype)), m_id(std::move(rh.m_id)), 
			isClosed(std::move(rh.isClosed))
		{}
		Curve& operator=(const Curve& lh)
		{
			if (this != std::addressof(lh))
			{
				m_pts = lh.m_pts;
				m_box = lh.m_box;
				m_gtype = lh.m_gtype;
				m_id = lh.m_id;
				isClosed = lh.isClosed;
			}
			return *this;
		}
		Curve& operator=(Curve&& rh) noexcept
		{
			if (this != std::addressof(rh))
			{
				m_pts = std::move(rh.m_pts);
				m_box = std::move(rh.m_box);
				m_gtype = std::move(rh.m_gtype);
				m_id = std::move(rh.m_id);
				isClosed = std::move(rh.isClosed);
			}
			return *this;
		}

		std::vector<cv::Point2d> m_pts;
		cv::Rect2d m_box;
		size_t m_gtype;
		size_t m_id;
		bool isClosed;
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

		cv::Mat draw(const DrawConfig& confi_draw, const DiscreConfig& confi_dis) const;

		std::vector<Curve> discretizeAll(const DiscreConfig& confi, cv::Rect2d& box, double shiftx = 0.0, double shifty = 0.0) const;

		cv::Size2d getScope() const;

	private:
		std::unique_ptr<Dxf2MatImpl> m_impl;
    };

}  // namespace libdxf2mat

#endif
