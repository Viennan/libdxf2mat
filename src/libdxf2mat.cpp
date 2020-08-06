#include "libdxf2mat.h"
#include "tinysplinecpp.h"
#include "dl_dxf.h"
#include "dl_creationadapter.h"

#include <cmath>
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm>

namespace libdxf2mat {

	using namespace std;
	using namespace cv;

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

	template<typename T>
	static inline Point round_point(const Point_<T>& p)
	{
		return Point(round(p.x), round(p.y));
	}

	static inline double to_rad(double an)
	{
		return an / 180.0 * CV_PI;
	}

	static inline double modular(double v, double mod)
	{
		return v - floor(v / mod) * mod;
	}

	// 2D transformation
	class Trans2D {
	public:
		Trans2D(): mat{1.0, 0.0, 0.0,
		               0.0, 1.0, 0.0}
		{}

		// zoom -> rotate -> translate
		Trans2D(double rad, double dx, double dy, double zoomx = 1.0, double zoomy = 1.0):
			mat{zoomx * cos(rad), -zoomy * sin(rad), dx, 
			    zoomx * sin(rad), zoomy * cos(rad), dy } 
		{}

		Trans2D(const Trans2D& trans) :
			mat{ trans.mat[0], trans.mat[1], trans.mat[2], 
			     trans.mat[3], trans.mat[4], trans.mat[5] } 
		{}

		Trans2D(double zoomx, double zoomy):
			mat{zoomx, 0.0, 0.0,
		        0.0, zoomy, 0.0}
		{}

		Trans2D& operator= (const Trans2D& lh) 
		{
			mat[0] = lh.mat[0];
			mat[1] = lh.mat[1];
			mat[2] = lh.mat[2];
			mat[3] = lh.mat[3];
			mat[4] = lh.mat[4];
			mat[5] = lh.mat[5];
			return *this;
		}

		Trans2D& operator*=(const Trans2D& rh)
		{
			double m0 = mat[0] * rh.mat[0] + mat[1] * rh.mat[3];
			double m1 = mat[0] * rh.mat[1] + mat[1] * rh.mat[4];
			double m2 = mat[0] * rh.mat[2] + mat[1] * rh.mat[5] + mat[2];

			double m3 = mat[3] * rh.mat[0] + mat[4] * rh.mat[3];
			double m4 = mat[3] * rh.mat[1] + mat[4] * rh.mat[4];
			double m5 = mat[3] * rh.mat[2] + mat[4] * rh.mat[5] + mat[5];

			mat[0] = m0, mat[1] = m1, mat[2] = m2;
			mat[3] = m3, mat[4] = m4, mat[5] = m5;
			return *this;
		}

		template<typename T>
		Point_<T> transfer(const Point_<T>& p) const
		{
			return Point_<T>(mat[0] * p.x + mat[1] * p.y + mat[2],
				mat[3] * p.x + mat[4] * p.y + mat[5]);
		}

		template<>
		Point transfer(const Point& p) const
		{
			return round_point(transfer(Point2d(p.x, p.y)));
		}

		double det() const
		{
			return mat[0] * mat[4] - mat[1] * mat[3];
		}

		double mat[6] = { 1.0, 0.0, 0.0, 
			              0.0 ,1.0, 0.0 };
	};

	static inline Trans2D operator*(const Trans2D& lh, const Trans2D& rh)
	{
		Trans2D tmp(lh);
		return tmp *= rh;
	}

	struct RasterizeData {
		RasterizeData(): pts(), isClosed(false) 
		{};
		RasterizeData(const vector<Point>& ps, bool flag) :
			pts(ps), isClosed(flag)
		{};
		RasterizeData(vector<Point>&& ps, bool flag) :
			pts(std::move(ps)), isClosed(flag)
		{};
		RasterizeData(const RasterizeData& lh) :
			pts(lh.pts), isClosed(lh.isClosed)
		{};
		RasterizeData(RasterizeData&& rh) noexcept:
		pts(std::move(rh.pts)), isClosed(std::move(rh.isClosed))
		{};
		RasterizeData& operator= (const RasterizeData& lh)
		{
			if (this != std::addressof(lh))
			{
				pts = lh.pts;
				isClosed = lh.isClosed;
			}
			return *this;
		}
		RasterizeData& operator= (RasterizeData&& rh) noexcept
		{
			if (this != std::addressof(rh))
			{
				pts = std::move(rh.pts);
				isClosed = std::move(rh.isClosed);
			}
			return *this;
		}

		vector<Point> pts;
		bool isClosed;
	};

	class Line {
	public:
		Line(const Point2d& from, const Point2d& to):
			m_pt1(from), m_pt2(to)
		{}

		RasterizeData discretize(double interval, const Trans2D& trans, int max_pts) const
		{
			RasterizeData ras_data;
			ras_data.isClosed = false;
			ras_data.pts.emplace_back(round_point(trans.transfer(m_pt1)));
			ras_data.pts.emplace_back(round_point(trans.transfer(m_pt2)));
			return ras_data;
		}
	    
		Point2d m_pt1;
		Point2d m_pt2;
	};

	// Use Micke's Formula to calculate center with begin point and end point and bulge for all cases.
	static inline double mickeFormula(const Vec3d& b, const Vec3d& e, Point2d& cen)
	{
		auto bulge = 0.5 * (1.0 / b[2] - b[2]);
		cen.x = 0.5 * ((b[0] + e[0]) - bulge * (e[1] - b[1]));
		cen.y = 0.5 * ((b[1] + e[1]) + bulge * (e[0] - b[0]));
		return sqrt(pow(b[0] - cen.x, 2) + pow(b[1] - cen.y, 2));
	}

	class Polylines {
	public:
		explicit Polylines(bool closed) :vertices(), isClosed(closed), 
			m_cirlen(0.0), isOpt(false)
		{}

		Polylines(const vector<Vec3d>& vs, bool closed):
			vertices(vs), isClosed(closed), m_cirlen(0.0), isOpt(false)
		{
			update();
		}

		void update()
		{
			// If it is a optimize polylines
			for (const auto& v : vertices)
			{
				if (v[2] != 0.0)
				{
					isOpt = true;
					break;
				}
			}
			if (isOpt)
			{
				// For optimize polylines
				size_t n = vertices.size();
				auto n_minus_1 = n - 1;
				for (size_t i = 0; i < n_minus_1; ++i)
				{
					if (vertices[i][2] == 0.0)
						continue;
					Point2d p;
					auto r = mickeFormula(vertices[i], vertices[i + 1], p);
					m_cirlen += 4.0 * r * abs(atan(vertices[i][2]));
				}
				if (isClosed && vertices[n - 1][2] != 0.0)
				{
					Point2d p;
					auto r = mickeFormula(vertices[n - 1], vertices[0], p);
					m_cirlen = 4.0 * r * abs(atan(vertices[n - 1][2]));
				}
			}
		}

		RasterizeData discretize(double interval, const Trans2D& trans, int max_pts) const
		{
			RasterizeData ras_data;
			ras_data.isClosed = isClosed;
			int freepts = max_pts < vertices.size() * 2 ? 0 : (max_pts - vertices.size());
			if (isOpt && freepts > 0)
			{
				freepts = ceil(min(double(freepts), m_cirlen * sqrt(trans.det()) / interval));
				size_t n = vertices.size();
				for (size_t i = 0; i < n; ++i)
				{
					const auto& vb = vertices[i];
					ras_data.pts.emplace_back(round_point(trans.transfer(Point2d(vb[0], vb[1]))));
					if (!isClosed && i == n - 1)
						break;
					const auto& ve = (i == n - 1) ? vertices[0] : vertices[i + 1];
					if (vb[2] != 0.0)
					{
						Point2d cen;
						auto r = mickeFormula(vb, ve, cen);
						auto cen_rad = 4.0 * atan(vb[2]);
						auto len = r * abs(cen_rad);
						int pt_nums = round(len / m_cirlen * freepts);
						if (pt_nums > 0)
						{
							double step = cen_rad / (pt_nums + 1);
							double theta = acos((vb[0] - cen.x) / r);
							theta = vb[1] < cen.y ? (2.0 * CV_PI - theta) : theta;
							theta += step;
							for (int j = 0; j < pt_nums; ++j)
							{
								Point2d p(r * cos(theta), r * sin(theta));
								p += cen;
								ras_data.pts.emplace_back(round_point(trans.transfer(p)));
								theta += step;
							}
						}
					}
				}
			}
			else
			{
				ras_data.pts.resize(vertices.size());
				std::transform(vertices.cbegin(), vertices.cend(), ras_data.pts.begin(),
					[&trans](const Vec3d& p) {
						return round_point(trans.transfer(Point2d(p[0], p[1])));
					});
			}
			return ras_data;
		}

		vector<Vec3d> vertices;
		bool isClosed;
	private:
		double m_cirlen;
		bool isOpt;
	};

	class Circle {
	public:
		Circle(const Point2d& cen, double r) : m_cen(cen), m_r(r) {}
		Circle(double x, double y, double r) : m_cen(x, y), m_r(r) {}

		RasterizeData discretize(double interval, const Trans2D& trans, int max_pts) const
		{
			// roughly estimates sample points number
			double zoom = sqrt(trans.det());
			double r = zoom * m_r;
			size_t pt_nums = max(1ull, size_t(min(double(max_pts - 1), ceil(2.0 * CV_PI * r / interval))));
			double step = 2 * CV_PI / pt_nums;

			double theta = 0.0;
			RasterizeData ras_data;
			ras_data.isClosed = true;
			ras_data.pts.resize(pt_nums);
			for (size_t i = 0; i < pt_nums; ++i)
			{
				Point2d pt(m_r * cos(theta), m_r * sin(theta));
				pt += m_cen;
				ras_data.pts[i] = round_point(trans.transfer(pt));
				theta += step;
			}
			return ras_data;
		}

		Point2d m_cen;
		double m_r;
	};

	class Arc {
	public:
		Arc(const Point2d& cen, double r, double rad_beg, double theta):
			m_cen(cen), m_r(r), m_beg(rad_beg), m_theta(theta)
		{}

		RasterizeData discretize(double interval, const Trans2D& trans, int max_pts) const
		{
			// roughly estimates sample points number
			double zoom = sqrt(trans.det());
			double r = zoom * m_r;
			size_t pt_nums = max(2ull, size_t(min(double(max_pts - 1), ceil(r * m_theta / interval))) + 1ull);
			double step = m_theta / (pt_nums - 1);

			double theta = m_beg;
			RasterizeData ras_data;
			ras_data.isClosed = false;
			ras_data.pts.resize(pt_nums);
			for (size_t i = 0; i < pt_nums; ++i)
			{
				Point2d pt(m_r * cos(theta), m_r * sin(theta));
				pt += m_cen;
				ras_data.pts[i] = round_point(trans.transfer(pt));
				theta += step;
			}
			return ras_data;
		}
	
		Point2d m_cen;
		double m_r;
		double m_beg;
		double m_theta;
	};

	class Ellipse {
	public:
		Ellipse(const Point2d& cen, const Size2d& axes, double rad, double beg, double theta):
			m_cen(cen), m_axes(axes), m_rot_rad(rad), m_beg(beg), m_theta(theta)
		{}

		RasterizeData discretize(double interval, const Trans2D& trans, int max_pts) const
		{
			// roughly estimates sample points number
			double zoom = sqrt(trans.det());
			double r = max(m_axes.width, m_axes.height) * zoom;
			size_t pt_nums = max(2ull, size_t(min(double(max_pts - 1), ceil(r * abs(m_theta) / interval))) + 1ull);
			double step = m_theta / (pt_nums - 1);

			double theta = m_beg;
			RasterizeData ras_data;
			ras_data.isClosed = false;
			ras_data.pts.resize(pt_nums);
			// Note: tran2d and trans have the same determinant.
			Trans2D tran2d = trans * Trans2D(m_rot_rad, m_cen.x, m_cen.y);
			for (int i = 0; i < pt_nums; ++i)
			{
				Point2d pt(cos(theta) * m_axes.width, sin(theta) * m_axes.height);
				ras_data.pts[i] = round_point(tran2d.transfer(pt));
				theta += step;
			}
			return ras_data;			
		}

		Point2d m_cen;
		Size2d m_axes;
		double m_rot_rad;
		double m_beg;
		double m_theta;
	};

	// A pipline between 2D nurbs to 3D b-spline
	class tinynurbs2d :public tinyspline::BSpline {

	public:
		tinynurbs2d(size_t nCtrlp, size_t deg = 3, tinyspline::BSpline::type type = TS_CLAMPED) :
			BSpline(nCtrlp, 3, deg, type) {}

		Point2d at(double u) const
		{
			auto val = BSpline::eval(u).result();
			return Point2d(val[0], val[1]);
		}

		void setNurbsControlPoints(const vector<Vec3d>& conPs)
		{
			vector<tinyspline::real> pts;
			for (const auto& p : conPs)
			{
				pts.push_back(p[0] * p[2]);
				pts.push_back(p[1] * p[2]);
				pts.push_back(p[2]);
			}
			BSpline::setControlPoints(pts);
		}

		vector<Point2d> sampleNurbs(size_t num) const
		{
			auto wPts = BSpline::sample(num);
			vector<Point2d> pts;
			for (int i = 0; i < wPts.size() - 2; i += 3)
			{
				double w = wPts[i + 2];
				if (w == 0.0)
					continue;

				pts.emplace_back(wPts[i] / w, wPts[i + 1] / w);
			}
			return pts;
		}

	private:
		using BSpline::setControlPoints;
		using BSpline::sample;

	};

	class Spline {
	public:
		Spline(unsigned int _degree, unsigned int vs, bool closed):
			degree(_degree), vertices(), knots(), isClosed(closed),
			m_impl(vs, degree), m_length(0.0)
		{}

		Spline(unsigned int _degree, const vector<Vec3d>& vs, const vector<double>& _knots, 
			bool closed, size_t _sampleNum) :
			degree(_degree), vertices(vs), knots(_knots), isClosed(closed), 
			m_impl(vertices.size(), degree), m_length(0.0)
		{
			update();
		}

		void update()
		{
			m_impl.setNurbsControlPoints(vertices);
			vector<tinyspline::real> ks;
			for (const auto& kn : knots)
				ks.push_back(kn);
			m_impl.setKnots(ks);

			// roughly estimates the length of spline
			auto pts = m_impl.sampleNurbs(100);
			int num = pts.size();
			for (size_t i = 1; i < num; ++i)
			{
				double dis = sqrt(pow(pts[i].x - pts[i - 1].x, 2) + pow(pts[i].y - pts[i - 1].y, 2));
				m_length += dis;
			}
			if (isClosed)
			{
				double dis = sqrt(pow(pts[num - 1].x - pts[0].x, 2) + pow(pts[num - 1].y - pts[0].y, 2));
				m_length += dis;
			}
		}

		RasterizeData discretize(double interval, const Trans2D& trans, int max_pts) const
		{
			// roughly estimates sample points number
			double len = m_length * sqrt(trans.det());
			size_t pt_nums = max(2ull, size_t(min(double(max_pts - 1), 
				ceil(len / interval))) + 1ull);
			auto pts = m_impl.sampleNurbs(pt_nums);
			RasterizeData ras_data;
			ras_data.isClosed = isClosed;
			ras_data.pts.resize(pts.size());
			std::transform(pts.cbegin(), pts.cend(), ras_data.pts.begin(),
				[&trans](const Point2d& p) { 
					return round_point(trans.transfer(p)); 
				});
			return ras_data;
		}

		unsigned int degree;
		vector<Vec3d> vertices;
		vector<double> knots;
		bool isClosed;

	private:
		tinynurbs2d m_impl;
		double m_length;
	};

	class InsertInfo {
	public:
		InsertInfo(const string& name, double x, double y, double sx, double sy, double rad, 
			int cs, int rs, double cSp, double rSp):
			m_name(name), trans2Ds(), id(numeric_limits<size_t>::max())
		{
			Trans2D trans(rad, x, y, sx, sy);
			for (size_t c = 0; c < cs; ++c)
			{
				for (size_t r = 0; r < rs; ++r)
					trans2Ds.push_back(Trans2D(0.0, c * cSp, r * rSp) * trans);
			}
		}

		/*! Name of the referred block. */
		string m_name;	
		size_t id;
		/*! transform matrixes from local to global */
		vector<Trans2D> trans2Ds;
		/*! Number of colums if we insert an array of the block or 1. */
		//int cols;
		/*! Number of rows if we insert an array of the block or 1. */
		//int rows;
		/*! Values for the spacing between cols. */
		//double colSp;
		/*! Values for the spacing between rows. */
		//double rowSp;

	};

	using GraphicIDs = array<vector<size_t>, GTYPES>;

	struct Block {
		Block(const string& name, size_t id):
			m_name(name), m_id(id), gIDs(), insertInfos()
		{}
		string m_name;
		size_t m_id;
		GraphicIDs gIDs;
		vector<InsertInfo> insertInfos;
	};

	struct GraphicAttr {
		size_t gtype;
		size_t id;
		Trans2D trans;
		//Rect2d box;
	};

	using GAttrs = array<vector<GraphicAttr>, GTYPES>;

	class Graphics {
	public:
		Graphics() = default;

		//@todo: caculate range of cad drawing from RasterizeData
		vector<RasterizeData> discretize(double interval, double dx, double dy, 
			double zoomx, double zoomy, int max_pts, Vec4i* range = nullptr) const
		{
			Trans2D trans(0.0, dx, dy, zoomx, zoomy);
			vector<RasterizeData> ras_data;
			for (const auto& g : m_attrs[LINE])
				ras_data.emplace_back(lines[g.id].discretize(interval, trans * g.trans, max_pts));
			for (const auto& g : m_attrs[POLYLINES])
				ras_data.emplace_back(polylines[g.id].discretize(interval, trans * g.trans, max_pts));
			for (const auto& g : m_attrs[CIRCLE])
				ras_data.emplace_back(circles[g.id].discretize(interval, trans * g.trans, max_pts));
			for (const auto& g : m_attrs[ARC])
				ras_data.emplace_back(arcs[g.id].discretize(interval, trans * g.trans, max_pts));
			for (const auto& g : m_attrs[ELLIPSE])
				ras_data.emplace_back(ellipses[g.id].discretize(interval, trans * g.trans, max_pts));
			for (const auto& g : m_attrs[SPLINE])
				ras_data.emplace_back(splines[g.id].discretize(interval, trans * g.trans, max_pts));

			if (range)
			{
				int minx = numeric_limits<int>::max(), miny = numeric_limits<int>::max();
				int maxx = numeric_limits<int>::min(), maxy = numeric_limits<int>::min();
				for (const auto& frag : ras_data)
				{
					for (const auto& p : frag.pts)
					{
						minx = min(minx, p.x);
						miny = min(miny, p.y);
						maxx = max(maxx, p.x);
						maxy = max(maxy, p.y);
					}
				}
				*range = Vec4i(minx, maxx, miny, maxy);
			}
			return ras_data;
		}

		void clear()
		{
			for (auto& a : m_attrs)
				a.clear();
			lines.clear();
			polylines.clear();
			circles.clear();
			arcs.clear();
			ellipses.clear();
			splines.clear();
		}
		
		GAttrs m_attrs;
		vector<Line> lines;
		vector<Polylines> polylines;
		vector<Circle> circles;
		vector<Arc> arcs;
		vector<Ellipse> ellipses;
		vector<Spline> splines;
		Point2d extMin, extMax;
	};

	// Interface to dxflib as official recommendation
	class DxfParser :public DL_CreationAdapter {
	public:
		explicit DxfParser(Graphics *ptr): g_ptr(ptr)
		{}

		// get range of cad drawing
		virtual void processCodeValuePair(unsigned int code, const std::string& name) override;

		// cache block
		virtual void addBlock(const DL_BlockData& dl) override;
		virtual void endBlock() override;

		virtual void addLine(const DL_LineData& dl) override;
		virtual void addArc(const DL_ArcData& dl) override;
		virtual void addCircle(const DL_CircleData& dl) override;
		virtual void addEllipse(const DL_EllipseData& dl) override;
		virtual void addPolyline(const DL_PolylineData& dl) override;
		virtual void addVertex(const DL_VertexData& dl) override;
		virtual void addSpline(const DL_SplineData& dl) override;
		virtual void addControlPoint(const DL_ControlPointData& dl) override;
		virtual void addKnot(const DL_KnotData& dl) override;
		virtual void addInsert(const DL_InsertData& dl) override;

		void postprocess();

	private:
		// parsed data
		Graphics *g_ptr = nullptr;

		// cache blocks
		unordered_map<string, size_t> m_name2id = {};
		vector<Block> blocks = { Block{"", 0} };
		size_t blockID = 0;

		// cache properties of polylines
		size_t poly_vnums = 0;

		// cache properties of spline
		size_t spline_vnums = 0;
		size_t spline_knums = 0;
		
		// used to get range of cad drawing
		string currentName = "";
		static const char extMin[8];
		static const char extMax[8];
		Point2d g_bl{ 0.0,0.0 }, g_tr{ 0.0,0.0 };
	};

	const char DxfParser::extMin[8] = "$EXTMIN";
	const char DxfParser::extMax[8] = "$EXTMAX";

	// get range of cad drawing
	void DxfParser::processCodeValuePair(unsigned int code, const std::string& name) 
	{
		if (code >= 0 && code <= 9)
		{
			if (name == extMin || name == extMax)
				currentName = name;
			else
			{
				currentName = "";
				return;
			}
		}

		if (currentName == extMin)
		{
			switch (code)
			{
			case 10:
				g_bl.x = stod(name);
				break;
			case 20:
				g_bl.y = stod(name);
				break;
			default:
				break;
			}
			return;
		}

		if (currentName == extMax)
		{
			switch (code)
			{
			case 10:
				g_tr.x = stod(name);
				break;
			case 20:
				g_tr.y = stod(name);
				break;
			default:
				break;
			}
			return;
		}
	}

	void DxfParser::addBlock(const DL_BlockData& dl)
	{
		blockID = blocks.size();
		blocks.emplace_back(dl.name, blockID);
		m_name2id[dl.name] = blockID;
	}

	void DxfParser::endBlock()
	{
		blockID = 0;
	}

	void DxfParser::addLine(const DL_LineData& dl)
	{
		auto& lines = g_ptr->lines;
		blocks[blockID].gIDs[LINE].push_back(lines.size());
		lines.emplace_back(Point2d(dl.x1, dl.y1), Point2d(dl.x2, dl.y2));
	}

	void DxfParser::addArc(const DL_ArcData& dl)
	{
		bool anti_clock = getExtrusion()->getDirection()[2] > 0.0;
		double rad1 = anti_clock ? dl.angle1 : -dl.angle1;
		double rad2 = anti_clock ? dl.angle2 : -dl.angle2;
		rad1 = modular(to_rad(rad1), 2 * CV_PI);
		rad2 = modular(to_rad(rad2), 2 * CV_PI);
		if ((rad1 == rad2 && dl.angle1 != dl.angle2) || rad1 > rad2)
			rad2 += 2 * CV_PI;
		double theta = rad2 - rad1;
		auto& arcs = g_ptr->arcs;
		blocks[blockID].gIDs[ARC].push_back(arcs.size());
		arcs.emplace_back(Point2d(dl.cx, dl.cy), dl.radius, 
			rad1, theta);
	}

	void DxfParser::addCircle(const DL_CircleData& dl)
	{
		auto& cirs = g_ptr->circles;
		blocks[blockID].gIDs[CIRCLE].push_back(cirs.size());
		cirs.emplace_back(Point2d(dl.cx, dl.cy), dl.radius);
	}

	void DxfParser::addEllipse(const DL_EllipseData& dl)
	{
		bool anti_clock = getExtrusion()->getDirection()[2] > 0.0;
		double rad1 = anti_clock ? dl.angle1 : -dl.angle1;
		double rad2 = anti_clock ? dl.angle2 : -dl.angle2;
		double theta = rad2 - rad1;
		if (theta > 2 * CV_PI)
			theta = modular(theta, 2 * CV_PI);
		if (theta < -2 * CV_PI)
			theta = modular(theta, -2 * CV_PI);
		double major = sqrt(dl.mx * dl.mx + dl.my * dl.my);
		double minor = major * dl.ratio;
		double rotate = acos(dl.mx / major);
		rotate = dl.my > 0 ? rotate : (2 * CV_PI - rotate);
		auto& elli = g_ptr->ellipses;
		blocks[blockID].gIDs[ELLIPSE].push_back(elli.size());
		elli.emplace_back(Point2d(dl.cx, dl.cy), Size2d(major, minor), rotate, rad1, theta);
	}

	void DxfParser::addPolyline(const DL_PolylineData& dl)
	{
		auto& polys = g_ptr->polylines;
		blocks[blockID].gIDs[POLYLINES].push_back(polys.size());
		polys.emplace_back(dl.flags & 0x01);
		poly_vnums = dl.number;
	}

	void DxfParser::addVertex(const DL_VertexData& dl)
	{
		auto& poly = g_ptr->polylines.back();
		poly.vertices.emplace_back(dl.x, dl.y, dl.bulge);
		if (poly.vertices.size() == poly_vnums)
			poly.update();
	}

	void DxfParser::addSpline(const DL_SplineData& dl)
	{
		auto& sps = g_ptr->splines;
		blocks[blockID].gIDs[SPLINE].push_back(sps.size());
		sps.emplace_back(dl.degree, dl.nControl, dl.flags & 0x01);
		spline_vnums = dl.nControl;
		spline_knums = dl.nKnots;
	}

	void DxfParser::addControlPoint(const DL_ControlPointData& dl)
	{
		auto& spline = g_ptr->splines.back();
		spline.vertices.emplace_back(dl.x, dl.y, dl.w);
		if (spline.vertices.size() == spline_vnums &&
			spline.knots.size() == spline_knums)
			spline.update();
	}

	void DxfParser::addKnot(const DL_KnotData& dl)
	{
		auto& spline = g_ptr->splines.back();
		spline.knots.emplace_back(dl.k);
		if (spline.vertices.size() == spline_vnums &&
			spline.knots.size() == spline_knums)
			spline.update();
	}

	void DxfParser::addInsert(const DL_InsertData& dl)
	{
		auto& insertInfos = blocks[blockID].insertInfos;
		insertInfos.emplace_back(dl.name, dl.ipx, dl.ipy, dl.sx, dl.sy,
			to_rad(dl.angle), dl.cols, dl.rows, dl.colSp, dl.rowSp);
	}

	void DxfParser::postprocess()
	{
		// fit inserts infos
		for (auto& block : blocks)
		{
			auto& inserts = block.insertInfos;
			auto iter = inserts.begin();
			while (iter != inserts.end())
			{
				if (m_name2id.count(iter->m_name))
				{
					iter->id = m_name2id[iter->m_name];
					++iter;
				}
				else
				{
					auto pos = iter - inserts.begin();
					std::swap(*iter, inserts.back());
					inserts.pop_back();
					iter = inserts.begin() + pos;
				}
			}
		}

		// Topological sort
		auto nb = blocks.size();
		vector<size_t> indegrees(nb, 0);
		for (const auto& b : blocks)
		{
			for (const auto& inser : b.insertInfos)
				++indegrees[inser.id];
		}
		queue<size_t> q;
		for (size_t i = 0; i < nb; ++i)
		{
			if (indegrees[i] == 0)
				q.push(i);
		}
		auto roots = q.size();
		vector<size_t> topo_ranks(nb);
		size_t topoid = 0;
		while (!q.empty())
		{
			auto t = q.front();
			q.pop();
			topo_ranks[topoid++] = t;
			for (const auto& inser : blocks[t].insertInfos)
			{
				if (--indegrees[inser.id] == 0)
					q.push(inser.id);
			}
		}
		std::reverse(topo_ranks.begin(), topo_ranks.end());

		// extract 
		vector<GAttrs> gAttrsColl(nb);
		for (auto id : topo_ranks)
		{
			const auto& b = blocks[id];
			auto& gAttrs = gAttrsColl[id];
			GraphicAttr attr;
			for (size_t gt = 0; gt < GTYPES; ++gt)
			{
				attr.gtype = gt;
				auto& gt_attrs = gAttrs[gt];
				for (auto gid : b.gIDs[gt])
				{
					attr.id = gid;
					gt_attrs.push_back(attr);
				}
			}
			for (const auto& inser : b.insertInfos)
			{
				const auto& inAttrs = gAttrsColl[inser.id];
				double xshift = 0.0;
				vector<Trans2D> trans2Ds;
				for (size_t gt = 0; gt < GTYPES; ++gt)
				{
					auto& gt_attrs = gAttrs[gt];
					for (const auto& inAttr : inAttrs[gt])
					{
						gt_attrs.push_back(inAttr);
						for (const auto& trans : inser.trans2Ds)
							gt_attrs.back().trans = trans * inAttr.trans;
					}
				}
			}
		}

		// merge roots
		for (size_t root = 0; root < roots; ++root)
		{
			const auto& mAttrs = gAttrsColl[topo_ranks[nb - root - 1]];
			for (size_t gt = 0; gt < GTYPES; ++gt)
			{
				auto& gt_attrs = g_ptr->m_attrs[gt];
				const auto& mt_attrs = mAttrs[gt];
				gt_attrs.insert(gt_attrs.end(), mt_attrs.begin(), mt_attrs.end());
			}
		}

		// assign drawing range
		g_ptr->extMin = g_bl;
		g_ptr->extMax = g_tr;
	}

	class Dxf2MatImpl {
	public:
		Dxf2MatImpl() = default;

		bool parse(const std::string& filename);
		cv::Mat draw(const DrawConfig& config) const;

		pair<Point2d, Point2d> getRange() const
		{
			return make_pair(m_graphics.extMin, m_graphics.extMax);
		}

	private:
		Graphics m_graphics;
	};

	bool Dxf2MatImpl::parse(const std::string& filename)
	{
		m_graphics.clear();
		auto dl_dxf = std::make_unique<DL_Dxf>();
		auto parser = std::make_unique<DxfParser>(&m_graphics);
		if(!dl_dxf->in(filename, parser.get()))
			return false;
		parser->postprocess();
		return true;
	}

	cv::Mat Dxf2MatImpl::draw(const DrawConfig& config) const
	{
		Vec4i range;
		auto ras_data = m_graphics.discretize(config.sample_interval, 0.0, 0.0, 
			config.zoom, config.zoom, config.max_sample_pts, &range);
		if (range[0] + config.maxSize.width - 1 < range[1] ||
			range[2] + config.maxSize.height - 1 < range[3])
			return cv::Mat();

		int width = range[1] - range[0] + 2 * config.margin.x + 1;
		int height = range[3] - range[2] + 2 * config.margin.y + 1;
		Mat canvas(height, width, config.mat_type, config.back_color);
		// convert to OpenCV coordinate
		int tx = range[0] - config.margin.x;
		int ty = range[3] + config.margin.y;
		for (auto& rpts : ras_data)
		{
			for (auto& p : rpts.pts)
			{
				p.x -= tx;
				p.y = ty - p.y;
			}
		}
		
		// draw
		for (const auto& rpts : ras_data)
		{
			polylines(canvas, rpts.pts, rpts.isClosed, 
				config.line_color, config.thickness, config.line_type);
		}

		return canvas;
	}

	Dxf2Mat::Dxf2Mat():
		m_impl(std::make_unique<Dxf2MatImpl>())
	{}

	Dxf2Mat::~Dxf2Mat()
	{}

	bool Dxf2Mat::parse(const std::string& filename)
	{
		return m_impl->parse(filename);
	}

	cv::Mat Dxf2Mat::draw(const DrawConfig& config) const
	{
		return m_impl->draw(config);
	}

	std::pair<cv::Point2d, cv::Point2d> Dxf2Mat::getRange() const
	{
		return m_impl->getRange();
	}

}  // namespace libdxf2mat


