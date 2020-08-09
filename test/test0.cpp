#include "libdxf2mat.h"
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	libdxf2mat::DrawConfig config;
	config.back_color = 0;
	config.line_color = 255;
	config.line_type = LINE_4;
	config.margin = Point(10, 10);
	config.mat_type = CV_8UC1;
	config.maxSize = { 2048, 2048 };
	config.thickness = 1;

	libdxf2mat::DiscreConfig conf_dis;
	conf_dis.max_sample_pts = 200;
	conf_dis.sample_interval = 3.0;
	conf_dis.zoom = 10.0;

	libdxf2mat::Dxf2Mat converter;
	if (!converter.parse("demo2000.dxf"))
	{
		cout << "fail to parse file" << endl;
		return 0;
	}
	auto canvas = converter.draw(config, conf_dis);
	if (canvas.empty())
	{
		cout << "fail to draw" << endl;
		return 0;
	}
	imshow("show", canvas);
	waitKey();
	return 0;
}