#include <chrono>
#include <iostream>

class Timer
{
private:
    bool   _running{true};
    bool   _delta_calculated{true};
    double _delta{0.};
    std::chrono::time_point<std::chrono::system_clock> _begin;
    std::chrono::time_point<std::chrono::system_clock> _end;
    std::ostream& _os;
    std::string _name;
public:

    Timer() : Timer("Timer", std::cout) {}

    Timer(const std::string& label) : Timer(label, std::cout) {}

    Timer(const std::string& label, std::ostream& outputstream) :
        _begin(std::chrono::system_clock::now()),
        _end(std::chrono::system_clock::now()),
        _os{outputstream},
        _name{label}
    { }

    ~Timer() { if(_running) stop(); }

    void start()
    {
        if (!_delta_calculated) 
        {
            const std::chrono::duration<double> DELTA = _end - _begin;
            _delta += DELTA.count();
            _delta_calculated = true;
        }

        _begin = std::chrono::system_clock::now();
        _end = std::chrono::system_clock::now();
        _running = true;
    }

    void stop()
    {
        _end = std::chrono::system_clock::now();
        _running = false;
        _delta_calculated = false;
    }

    void reset()
    {
        _running = false;
        _delta_calculated = true;
        _delta = 0;
    }

    double elapsed()
    {
        if (_running) stop();

        if (!_delta_calculated)
        {
            const std::chrono::duration<double> delta = _end - _begin;
            _delta += delta.count();
            _delta_calculated = true;
        }

        return _delta;
    }

    void print() { _os << "# "<< _name <<" : " << elapsed()  << "s\n"; }
};