#include <bits/stdc++.h>
using namespace std;    

// DFS não recursiva
void dfs_non_recursive(int start, const vector<vector<int>>& adjList) {
    vector<bool> visited(adjList.size(), false);
    stack<int> s;

    s.push(start);

    while(!s.empty()) {
        int current = s.top();
        s.pop();

        if (!visited[current]) {
            visited[current] = true;
            cout << current << " ";

            // Para garantir ordem semelhante à recursiva, adiciona os vizinhos de trás pra frente
            for (auto it = adjList[current].rbegin(); it != adjList[current].rend(); ++it) {
                if (!visited[*it]) {
                    s.push(*it);
                }
            }
        }
    }
}

// DFS recursiva
void dfs_recursive(int start, const vector<vector<int>>& adjList, vector<bool>& visited) {
    visited[start] = true;
    cout << start << " ";

    for(auto x: adjList[start]) {
        if(!visited[x]) {
            dfs_recursive(x, adjList, visited);
        }   
    }
}

int main() {
    // Grafo representado por lista de adjacência
    vector<vector<int>> adjList = {
        {1, 2},    // 0
        {0, 3, 4}, // 1
        {0, 5},    // 2
        {1},       // 3
        {1, 6, 7}, // 4
        {2, 8},    // 5
        {4},       // 6
        {4, 9},    // 7
        {5},       // 8
        {7}        // 9
    };

    int startVertex = 0;

    cout << "DFS não recursiva: ";
    dfs_non_recursive(startVertex, adjList);
    cout << endl;

    cout << "DFS recursiva: ";
    vector<bool> visited(adjList.size(), false); // vetor reiniciado
    dfs_recursive(startVertex, adjList, visited);
    cout << endl;

    return 0;
}
